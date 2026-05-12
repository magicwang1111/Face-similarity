from __future__ import annotations

import os
import re
from pathlib import Path


_WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")
_WINDOWS_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def normalize_platform_path_text(value: str | Path, platform_name: str | None = None) -> str:
    text = str(value)
    platform_name = os.name if platform_name is None else platform_name
    wsl_match = _WSL_MOUNT_RE.match(text)
    if platform_name == "nt" and wsl_match:
        drive = wsl_match.group(1).upper()
        remainder = (wsl_match.group(2) or "").replace("/", "\\")
        return f"{drive}:\\{remainder}" if remainder else f"{drive}:\\"

    windows_match = _WINDOWS_DRIVE_RE.match(text)
    if platform_name != "nt" and windows_match:
        drive = windows_match.group(1).lower()
        remainder = windows_match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{remainder}"

    return text


def normalize_platform_path(value: str | Path) -> Path:
    return Path(normalize_platform_path_text(value)).expanduser()
