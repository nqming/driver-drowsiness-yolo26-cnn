/**
 * Logic điều khiển Buồng lái Ô tô Thông minh (frontend/app.js)
 * - Kết nối WebSocket thời gian thực (/ws/telemetry) kèm dự phòng HTTP Polling
 * - Tích hợp Web Audio API phát còi báo động khẩn cấp Mức 2
 * - Tích hợp Web Speech API (speechSynthesis, ngôn ngữ vi-VN) đọc cảnh báo tiếng Việt
 * - Vẽ biểu đồ đường thời gian thực 30 giây bằng Chart.js
 */

(() => {
  // ==================== TRẠNG THÁI ỨNG DỤNG ====================
  let audioEnabled = true;
  let audioCtx = null;
  let sirenInterval = null;
  let sirenPlaying = false;
  let lastSpeechTime = 0;
  let lastAlertLevel = 0;

  let currentGear = "D";
  let currentSpeed = 60;
  let currentThreshold = 2.0;
  let currentIrMode = false;

  let ws = null;
  let lastChartPushTs = 0;
  let lastLogsFetchTs = 0;
  let cachedFigures = [];

  // ==================== KHỞI TẠO WEB AUDIO API & WEB SPEECH API ====================
  function ensureAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
  }

  function playSirenPulse() {
    if (!audioEnabled) return;
    ensureAudioContext();
    if (!audioCtx) return;

    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(860, now);
    osc.frequency.linearRampToValueAtTime(1240, now + 0.22);
    osc.frequency.linearRampToValueAtTime(860, now + 0.44);

    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.04);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.46);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(now);
    osc.stop(now + 0.47);
  }

  function setSirenState(shouldPlay) {
    if (shouldPlay && audioEnabled) {
      if (!sirenPlaying) {
        sirenPlaying = true;
        playSirenPulse();
        sirenInterval = setInterval(playSirenPulse, 520);
      }
    } else if (sirenPlaying) {
      sirenPlaying = false;
      if (sirenInterval) {
        clearInterval(sirenInterval);
        sirenInterval = null;
      }
    }
  }

  function speakVietnamese(message, urgent = false) {
    if (!audioEnabled || !message || !("speechSynthesis" in window)) return;
    const now = Date.now();
    const cooldown = urgent ? 6000 : 7500;
    if (now - lastSpeechTime < cooldown) return;

    lastSpeechTime = now;
    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(message);
      utter.lang = "vi-VN";
      utter.rate = urgent ? 1.08 : 1.0;
      utter.pitch = urgent ? 1.08 : 1.0;

      const voices = window.speechSynthesis.getVoices() || [];
      const viVoice = voices.find((v) => v.lang && v.lang.toLowerCase().includes("vi"));
      if (viVoice) {
        utter.voice = viVoice;
      }
      window.speechSynthesis.speak(utter);
    } catch (_) {}
  }

  // Mở khóa AudioContext ngay lần tương tác đầu tiên của người dùng
  document.addEventListener("click", () => ensureAudioContext(), { once: true });

  // ==================== KHỞI TẠO BIỂU ĐỒ CHART.JS 30 GIÂY ====================
  const MAX_POINTS = 60; // 60 điểm * 0.5s = 30 giây gần nhất
  const chartLabels = Array(MAX_POINTS).fill("");
  const dataClosed = Array(MAX_POINTS).fill(0);
  const dataYawn = Array(MAX_POINTS).fill(0);

  const ctxChart = document.getElementById("realtimeChart").getContext("2d");
  const realtimeChart = new Chart(ctxChart, {
    type: "line",
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: "Xác suất Nhắm mắt P(Nhắm mắt) %",
          data: dataClosed,
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.14)",
          borderWidth: 2.2,
          fill: true,
          tension: 0.32,
          pointRadius: 0,
        },
        {
          label: "Xác suất Ngáp P(Ngáp) %",
          data: dataYawn,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.10)",
          borderWidth: 2.0,
          fill: true,
          tension: 0.32,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#cbd5e1", font: { family: "Inter", size: 11, weight: "600" } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 8, font: { size: 10 } },
          grid: { color: "rgba(51, 65, 85, 0.25)" },
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            color: "#94a3b8",
            stepSize: 25,
            callback: (val) => `${val}%`,
            font: { size: 10 },
          },
          grid: { color: "rgba(51, 65, 85, 0.35)" },
        },
      },
    },
  });

  // ==================== CẬP NHẬT GIAO DIỆN TỪ TELEMETRY ====================
  function updateDashboardUI(t) {
    if (!t) return;

    document.getElementById("hudFps").textContent = `${(t.fps || 0).toFixed(1)} FPS`;
    document.getElementById("hudLatency").textContent = `${(t.latency_ms || 0).toFixed(1)} ms`;
    document.getElementById("hudPitch").textContent = `${(t.head_pitch || 0).toFixed(1)}°`;

    document.getElementById("statYoloConf").textContent = t.driver_detected
      ? `Tài xế (${(t.yolo_conf || 92).toFixed(0)}%)`
      : (t.active ? "Đang quét..." : "Tạm dừng");
    document.getElementById("statEar").textContent = (t.ear || 0).toFixed(3);
    document.getElementById("statMar").textContent = (t.mar || 0).toFixed(3);
    document.getElementById("statClosedSec").textContent = `${(t.eye_closed_duration || 0).toFixed(2)} giây`;
    document.getElementById("statYawnCount").textContent = `${t.yawns_per_min || 0} lần/phút`;
    document.getElementById("landmarkBackendTag").textContent = t.landmark_backend || "MediaPipe FaceMesh";

    // Giai đoạn 2: ROI & thanh xác suất CNN
    if (t.roi_left_eye) document.getElementById("roiLeftEye").src = t.roi_left_eye;
    if (t.roi_right_eye) document.getElementById("roiRightEye").src = t.roi_right_eye;
    if (t.roi_mouth) document.getElementById("roiMouth").src = t.roi_mouth;

    const pClosed = Math.min(100, Math.max(0, t.p_closed || 0));
    const pYawn = Math.min(100, Math.max(0, t.p_yawn || 0));
    document.getElementById("txtPClosed").textContent = `${pClosed.toFixed(1)}%`;
    document.getElementById("barPClosed").style.width = `${pClosed}%`;
    document.getElementById("txtPYawn").textContent = `${pYawn.toFixed(1)}%`;
    document.getElementById("barPYawn").style.width = `${pYawn}%`;
    document.getElementById("txtYawnDur").textContent = `Ngáp liên tục: ${(t.yawn_duration || 0).toFixed(2)}s`;

    // Huy hiệu trạng thái mô hình
    const badgeCnn = document.getElementById("badgeCnn");
    if (t.cnn_ready) {
      badgeCnn.textContent = "● CNN Đa Dữ Liệu: Đã nạp (MobileNetV3)";
      badgeCnn.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-violet-500/20 text-violet-300 border border-violet-500/40";
    } else {
      badgeCnn.textContent = "● CNN: Chế độ dự phòng EAR/MAR";
      badgeCnn.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40";
    }

    // Huy hiệu Hồng ngoại Ban đêm
    const irOverlay = document.getElementById("irBadgeOverlay");
    if (t.ir_mode) {
      irOverlay.classList.remove("hidden");
    } else {
      irOverlay.classList.add("hidden");
    }

    // Băng rôn cảnh báo đa tầng & Ngữ cảnh Xe đỗ
    const banner = document.getElementById("globalAlertBanner");
    const dot = document.getElementById("alertStatusDot");
    const titleEl = document.getElementById("alertTitleText");
    const subEl = document.getElementById("alertSubtitleText");

    titleEl.textContent = t.alert_title || "HỆ THỐNG SẴN SÀNG";
    subEl.textContent = t.alert_subtitle || "";

    banner.classList.remove(
      "alert-danger-flash",
      "border-l-emerald-500",
      "border-l-amber-500",
      "border-l-rose-500",
      "border-l-slate-400"
    );

    if (t.standby_mode) {
      banner.classList.add("border-l-slate-400");
      dot.className = "w-4 h-4 rounded-full bg-slate-400";
      titleEl.className = "text-base md:text-lg font-extrabold tracking-wide text-slate-200";
      setSirenState(false);
    } else if (t.alert_level === 2) {
      banner.classList.add("border-l-rose-500", "alert-danger-flash");
      dot.className = "w-4 h-4 rounded-full bg-rose-500 shadow-lg shadow-rose-500/80";
      titleEl.className = "text-base md:text-lg font-extrabold tracking-wide text-rose-400";
      setSirenState(Boolean(t.siren_active));
      if (t.voice_message) speakVietnamese(t.voice_message, true);
      // Tự động làm mới bảng vi phạm khi có cảnh báo Mức 2
      if (Date.now() - lastLogsFetchTs > 2500) {
        fetchViolationLogs();
      }
    } else if (t.alert_level === 1) {
      banner.classList.add("border-l-amber-500");
      dot.className = "w-4 h-4 rounded-full bg-amber-400 shadow-lg shadow-amber-400/60";
      titleEl.className = "text-base md:text-lg font-extrabold tracking-wide text-amber-300";
      setSirenState(false);
      if (t.voice_message) speakVietnamese(t.voice_message, false);
    } else {
      banner.classList.add("border-l-emerald-500");
      dot.className = "w-4 h-4 rounded-full bg-emerald-400 shadow-lg shadow-emerald-500/50";
      titleEl.className = "text-base md:text-lg font-extrabold tracking-wide text-emerald-400";
      setSirenState(false);
    }

    lastAlertLevel = t.alert_level || 0;

    // Đẩy dữ liệu vào biểu đồ 30s mỗi 500ms
    const nowMs = Date.now();
    if (t.active && nowMs - lastChartPushTs >= 500) {
      lastChartPushTs = nowMs;
      chartLabels.push(t.timestamp || new Date().toLocaleTimeString("vi-VN"));
      chartLabels.shift();
      dataClosed.push(pClosed);
      dataClosed.shift();
      dataYawn.push(pYawn);
      dataYawn.shift();
      realtimeChart.update("none");
    }
  }

  // ==================== KẾT NỐI WEBSOCKET & DỰ PHÒNG POLLING ====================
  function connectTelemetrySocket() {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${window.location.host}/ws/telemetry`);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        updateDashboardUI(data);
      } catch (_) {}
    };
    ws.onclose = () => {
      setTimeout(connectTelemetrySocket, 1800);
    };
  }

  // ==================== ĐIỀU KHIỂN XE & NGỮ CẢNH ĐỖ XE ====================
  function refreshVehicleContextUI() {
    document.getElementById("speedDisplay").textContent = `${currentSpeed}`;
    document.getElementById("speedGaugeBar").style.width = `${(currentSpeed / 140) * 100}%`;
    document.getElementById("thresholdValueText").textContent = `${currentThreshold.toFixed(1)} giây`;

    document.querySelectorAll(".gear-btn").forEach((btn) => {
      const g = btn.getAttribute("data-gear");
      btn.classList.remove("active-P", "active-N", "active-D");
      if (g === currentGear) {
        btn.classList.add(`active-${g}`);
      }
    });

    const isParkedOrSlow = currentGear === "P" || currentSpeed <= 5;
    const isNeutral = currentGear === "N" && !isParkedOrSlow;
    const notice = document.getElementById("parkedContextNotice");
    const badge = document.getElementById("contextBadge");
    const gearText = document.getElementById("gearExplainText");

    if (isParkedOrSlow) {
      notice.classList.remove("hidden");
      badge.textContent = "CHẾ ĐỘ CHỜ (XE ĐỖ)";
      badge.className = "px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-slate-700 text-slate-200 border border-slate-500";
      gearText.textContent = currentGear === "P" ? "P — Đang đỗ xe (Ngắt còi)" : `${currentGear} — Vận tốc ≤ 5 km/h (Ngắt còi)`;
      gearText.className = "text-amber-300";
      setSirenState(false);
    } else if (isNeutral) {
      notice.classList.remove("hidden");
      badge.textContent = "SỐ MO (TẠM CHỜ)";
      badge.className = "px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40";
      gearText.textContent = "N — Số mo (Tạm ngắt còi cảnh báo)";
      gearText.className = "text-amber-300";
      setSirenState(false);
    } else {
      notice.classList.add("hidden");
      badge.textContent = "GIÁM SÁT CHỦ ĐỘNG";
      badge.className = "px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
      gearText.textContent = "D — Chế độ Lái xe (Bật cảnh báo 3 mức)";
      gearText.className = "text-emerald-400";
    }
  }

  async function syncVehicleToBackend() {
    refreshVehicleContextUI();
    try {
      await fetch("/api/vehicle/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speed_kmh: currentSpeed,
          gear: currentGear,
          drowsy_threshold_sec: currentThreshold,
          ir_mode: currentIrMode,
        }),
      });
    } catch (_) {}
  }

  document.getElementById("sliderSpeed").addEventListener("input", (e) => {
    currentSpeed = parseInt(e.target.value, 10) || 0;
    syncVehicleToBackend();
  });

  document.getElementById("sliderThreshold").addEventListener("input", (e) => {
    currentThreshold = parseFloat(e.target.value) || 2.0;
    syncVehicleToBackend();
  });

  document.querySelectorAll(".gear-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentGear = btn.getAttribute("data-gear") || "D";
      if (currentGear === "P") {
        currentSpeed = 0;
        document.getElementById("sliderSpeed").value = "0";
      } else if (currentGear === "D" && currentSpeed === 0) {
        currentSpeed = 60;
        document.getElementById("sliderSpeed").value = "60";
      }
      syncVehicleToBackend();
    });
  });

  // ==================== ĐIỀU KHIỂN CAMERA & CHẾ ĐỘ HỒNG NGOẠI ====================
  function refreshVideoElement() {
    const img = document.getElementById("videoStream");
    img.src = `/api/video_feed?t=${Date.now()}`;
  }

  document.getElementById("btnStartWebcam").addEventListener("click", async () => {
    ensureAudioContext();
    await fetch("/api/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "webcam", ir_mode: currentIrMode }),
    });
    refreshVideoElement();
  });

  document.getElementById("btnStartDemo").addEventListener("click", async () => {
    ensureAudioContext();
    await fetch("/api/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "demo", ir_mode: currentIrMode }),
    });
    refreshVideoElement();
  });

  document.getElementById("btnStopStream").addEventListener("click", async () => {
    setSirenState(false);
    await fetch("/api/camera/stop", { method: "POST" });
    refreshVideoElement();
  });

  document.getElementById("inputVideoUpload").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    ensureAudioContext();
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch("/api/camera/upload_video", {
      method: "POST",
      body: formData,
    });
    if (resp.ok) {
      refreshVideoElement();
    } else {
      alert("Không thể mở tệp video đã chọn. Vui lòng chọn tệp .mp4 hoặc .avi hợp lệ!");
    }
    e.target.value = "";
  });

  document.getElementById("btnToggleIR").addEventListener("click", () => {
    currentIrMode = !currentIrMode;
    const btn = document.getElementById("btnToggleIR");
    const lbl = document.getElementById("irStatusLabel");
    if (currentIrMode) {
      lbl.textContent = "Hồng ngoại Đêm (IR): BẬT";
      btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-purple-700 hover:bg-purple-600 text-white border border-purple-400 transition flex items-center gap-1.5 shadow";
    } else {
      lbl.textContent = "Hồng ngoại Đêm (IR): TẮT";
      btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-600 transition flex items-center gap-1.5";
    }
    syncVehicleToBackend();
  });

  // Nút Bật/Tắt Âm thanh & Nút Thử Còi + Giọng nói Tiếng Việt
  document.getElementById("btnToggleAudio").addEventListener("click", () => {
    audioEnabled = !audioEnabled;
    ensureAudioContext();
    document.getElementById("audioIcon").textContent = audioEnabled ? "🔊" : "🔇";
    document.getElementById("audioText").textContent = audioEnabled
      ? "Âm thanh & Giọng nói: BẬT"
      : "Âm thanh & Giọng nói: TẮT";
    document.getElementById("btnToggleAudio").className = audioEnabled
      ? "px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition flex items-center gap-1.5 shadow"
      : "px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-slate-300 transition flex items-center gap-1.5";
    if (!audioEnabled) {
      setSirenState(false);
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    }
  });

  document.getElementById("btnTestVoice").addEventListener("click", () => {
    audioEnabled = true;
    ensureAudioContext();
    playSirenPulse();
    lastSpeechTime = 0;
    speakVietnamese("Hệ thống giám sát buồng lái thông minh đã sẵn sàng. Chúc bạn lái xe an toàn!", false);
  });

  // ==================== THÔNG TIN MÔ HÌNH, BIỂU ĐỒ & LỊCH SỬ VI PHẠM ====================
  async function loadModelAndDatasetInfo() {
    try {
      const resp = await fetch("/api/model_info");
      if (!resp.ok) return;
      const info = await resp.json();

      if (info.dataset_summary && info.dataset_summary.total_images) {
        document.getElementById("infoTotalImages").textContent = info.dataset_summary.total_images.toLocaleString("vi-VN");
      }
      if (info.training_metrics && info.training_metrics.eval_summary) {
        const ev = info.training_metrics.eval_summary;
        if (ev.Dataset1_YawnEye) {
          document.getElementById("accDs1").textContent = `${ev.Dataset1_YawnEye.accuracy.toFixed(1)}%`;
        }
        if (ev.Dataset2_MRLEye) {
          document.getElementById("accDs2").textContent = `${ev.Dataset2_MRLEye.accuracy.toFixed(1)}%`;
        }
        if (ev.Combined_All) {
          document.getElementById("accCombined").textContent = `${ev.Combined_All.accuracy.toFixed(1)}%`;
        }
      }
      cachedFigures = info.figures || [];
    } catch (_) {}
  }

  async function fetchViolationLogs() {
    lastLogsFetchTs = Date.now();
    try {
      const resp = await fetch("/api/logs?limit=30");
      if (!resp.ok) return;
      const data = await resp.json();
      document.getElementById("totalViolationsCount").textContent =
        data.stats && typeof data.stats.total_violations === "number" ? data.stats.total_violations : 0;

      const tbody = document.getElementById("logsTableBody");
      const items = data.items || [];
      if (items.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="9" class="py-6 text-center text-slate-500">
              Chưa có vi phạm ngủ gật nào được ghi nhận trong phiên làm việc.
            </td>
          </tr>`;
        return;
      }

      tbody.innerHTML = items
        .map((row) => {
          const modeBadge = row.ir_mode
            ? `<span class="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-semibold">🌙 Hồng ngoại</span>`
            : `<span class="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 font-semibold">☀️ RGB Ngày</span>`;
          const snapCell = row.snapshot_url
            ? `<button data-snap="${row.snapshot_url}" data-time="${row.timestamp}" class="btn-view-snap inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-sky-300 font-semibold">
                 <img src="${row.snapshot_url}" alt="Snap" class="w-9 h-7 object-cover rounded" />
                 Xem ảnh
               </button>`
            : `<span class="text-slate-500">--</span>`;

          return `
            <tr class="hover:bg-slate-900/80 transition">
              <td class="py-2.5 px-3 mono-num text-slate-400">#${row.id}</td>
              <td class="py-2.5 px-3 mono-num text-slate-200 font-medium">${row.timestamp}</td>
              <td class="py-2.5 px-3">
                <span class="px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">
                  🚨 ${row.alert_label}
                </span>
              </td>
              <td class="py-2.5 px-3 text-slate-300">${row.violation_reason}</td>
              <td class="py-2.5 px-3 mono-num font-bold text-white">${row.speed_kmh} km/h (${row.gear})</td>
              <td class="py-2.5 px-3 mono-num">
                <span class="text-rose-400 font-bold">${row.p_closed}%</span> /
                <span class="text-amber-400">${row.p_yawn}%</span>
              </td>
              <td class="py-2.5 px-3 mono-num text-slate-300">${row.head_pitch}°</td>
              <td class="py-2.5 px-3">${modeBadge}</td>
              <td class="py-2.5 px-3 text-center">${snapCell}</td>
            </tr>`;
        })
        .join("");

      document.querySelectorAll(".btn-view-snap").forEach((btn) => {
        btn.addEventListener("click", () => {
          document.getElementById("snapshotModalImg").src = btn.getAttribute("data-snap");
          document.getElementById("snapshotModalTitle").textContent =
            `Ảnh chụp hiện trường vi phạm — ${btn.getAttribute("data-time")}`;
          document.getElementById("snapshotModal").classList.remove("hidden");
        });
      });
    } catch (_) {}
  }

  document.getElementById("btnRefreshLogs").addEventListener("click", fetchViolationLogs);
  document.getElementById("btnClearLogs").addEventListener("click", async () => {
    await fetch("/api/logs", { method: "DELETE" });
    fetchViolationLogs();
  });

  // Lưu cấu hình Telegram Bot
  document.getElementById("btnSaveTelegram").addEventListener("click", async () => {
    const token = document.getElementById("inputTgToken").value.trim();
    const chatId = document.getElementById("inputTgChatId").value.trim();
    const resp = await fetch("/api/telegram/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bot_token: token, chat_id: chatId, enabled: Boolean(token && chatId) }),
    });
    if (resp.ok) {
      const st = await resp.json();
      document.getElementById("tgStatusText").textContent = st.last_status;
      document.getElementById("tgStatusText").className = st.enabled
        ? "text-[11px] text-emerald-400 font-semibold"
        : "text-[11px] text-slate-400";
    }
  });

  // Modal Biểu đồ học thuật 300 DPI
  document.getElementById("btnOpenCharts").addEventListener("click", async () => {
    await loadModelAndDatasetInfo();
    const gallery = document.getElementById("chartsGalleryBody");
    if (!cachedFigures.length) {
      gallery.innerHTML = `<div class="col-span-2 text-center py-8 text-slate-400">Đang tạo biểu đồ đánh giá trong thư mục reports/figures/...</div>`;
    } else {
      gallery.innerHTML = cachedFigures
        .map(
          (fg) => `
          <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-2">
            <div class="text-xs font-bold text-sky-300">${fg.title}</div>
            <a href="${fg.url}" target="_blank" title="Nhấn để mở ảnh gốc 300 DPI">
              <img src="${fg.url}?t=${Date.now()}" alt="${fg.title}" class="w-full rounded-lg border border-slate-800 bg-white" />
            </a>
          </div>`
        )
        .join("");
    }
    document.getElementById("chartsModal").classList.remove("hidden");
  });

  document.getElementById("btnCloseChartsModal").addEventListener("click", () => {
    document.getElementById("chartsModal").classList.add("hidden");
  });
  document.getElementById("btnCloseSnapshotModal").addEventListener("click", () => {
    document.getElementById("snapshotModal").classList.add("hidden");
  });

  // ==================== KHỞI CHẠY BAN ĐẦU ====================
  refreshVehicleContextUI();
  connectTelemetrySocket();
  loadModelAndDatasetInfo();
  fetchViolationLogs();
  setInterval(fetchViolationLogs, 6000);
  setInterval(loadModelAndDatasetInfo, 15000);
})();
