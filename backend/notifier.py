"""
Module gửi cảnh báo khẩn cấp qua Telegram Bot (backend/notifier.py)
khi phát hiện vi phạm Mức 2 (NGUY HIỂM: TÀI XẾ NGỦ GẬT!).
Hoạt động bất đồng bộ trên luồng nền (Zero-Crash, không gây trễ luồng video thời gian thực).
"""

import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

import requests


class TelegramNotifier:
    """Quản lý cấu hình và gửi cảnh báo kèm ảnh chụp hiện trường qua Telegram Bot API."""

    def __init__(self) -> None:
        self.bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled: bool = bool(self.bot_token and self.chat_id)
        self.last_status: str = "Sẵn sàng (Chưa cấu hình Token)" if not self.enabled else "Đã kết nối Telegram Bot"

    def configure(self, bot_token: str, chat_id: str, enabled: bool = True) -> Dict:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.enabled = bool(enabled and self.bot_token and self.chat_id)
        self.last_status = (
            "Đã kích hoạt cảnh báo Telegram"
            if self.enabled
            else "Tạm tắt (Chưa nhập Bot Token / Chat ID)"
        )
        return self.get_status()

    def get_status(self) -> Dict:
        masked_token = (
            f"{self.bot_token[:6]}...{self.bot_token[-4:]}"
            if len(self.bot_token) > 12
            else ("Đã nhập" if self.bot_token else "")
        )
        return {
            "enabled": self.enabled,
            "configured": bool(self.bot_token and self.chat_id),
            "masked_token": masked_token,
            "chat_id": self.chat_id,
            "last_status": self.last_status,
        }

    def send_drowsiness_alert_async(
        self,
        caption_info: Dict,
        snapshot_file: Optional[Path] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Gửi thông báo và ảnh chụp trên luồng nền để không làm nghẽn pipeline AI."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            if on_complete:
                on_complete(False)
            return

        thread = threading.Thread(
            target=self._send_sync,
            args=(caption_info, snapshot_file, on_complete),
            daemon=True,
        )
        thread.start()

    def _send_sync(
        self,
        info: Dict,
        snapshot_file: Optional[Path],
        on_complete: Optional[Callable[[bool], None]],
    ) -> None:
        caption = (
            "🚨 *CẢNH BÁO KHẨN CẤP: PHÁT HIỆN TÀI XẾ NGỦ GẬT!*\n"
            f"• *Thời gian:* `{info.get('timestamp', '')}`\n"
            f"• *Nguyên nhân:* {info.get('reason', '')}\n"
            f"• *Vận tốc xe:* `{info.get('speed_kmh', 0):.0f} km/h` (Số `{info.get('gear', 'D')}`)\n"
            f"• *CNN P(Nhắm mắt):* `{info.get('p_closed', 0):.1f}%` | *P(Ngáp):* `{info.get('p_yawn', 0):.1f}%`\n"
            f"• *Góc cúi đầu (Pitch):* `{info.get('head_pitch', 0):.1f}°`\n"
            f"• *Chế độ Camera:* {'Hồng ngoại Ban đêm (NIR)' if info.get('ir_mode') else 'RGB Ban ngày'}"
        )

        success = False
        try:
            base_url = f"https://api.telegram.org/bot{self.bot_token}"
            if snapshot_file is not None and snapshot_file.exists():
                with open(snapshot_file, "rb") as photo_fp:
                    resp = requests.post(
                        f"{base_url}/sendPhoto",
                        data={
                            "chat_id": self.chat_id,
                            "caption": caption,
                            "parse_mode": "Markdown",
                        },
                        files={"photo": photo_fp},
                        timeout=8,
                    )
                success = resp.status_code == 200
            else:
                resp = requests.post(
                    f"{base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": caption,
                        "parse_mode": "Markdown",
                    },
                    timeout=8,
                )
                success = resp.status_code == 200

            self.last_status = "Gửi cảnh báo thành công" if success else f"Lỗi Telegram HTTP {resp.status_code}"
        except Exception as exc:
            self.last_status = f"Không thể kết nối Telegram: {type(exc).__name__}"
            success = False

        if on_complete:
            try:
                on_complete(success)
            except Exception:
                pass
