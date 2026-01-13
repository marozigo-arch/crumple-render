from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderParams:
    poster_interp: str = "Linear"  # Linear|Cubic|Closest
    light_scale: float = 3.0


def _interp_flag(name: str) -> int:
    if name == "Closest":
        return cv2.INTER_NEAREST
    if name == "Cubic":
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * (x ** (1.0 / 2.4)) - a).astype(np.float32)


def _read_u16(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG: {path}")
    return img


def _read_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def decode_uv_to_mapxy(uv_u16: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode uv_{frame}.png (uint16) into OpenCV remap matrices (float32 map_x/map_y).
    Convention matches flow_assets.flow_to_uv16() for uv_mode='w_minus_1'.
    """
    u = uv_u16[:, :, 0].astype(np.float32) / 65535.0
    v = uv_u16[:, :, 1].astype(np.float32) / 65535.0
    map_x = u * float(w - 1)
    map_y = (1.0 - v) * float(h - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def render_sequence(
    poster_path: Path,
    assets_dir: Path,
    out_dir: Path,
    frame_start: int,
    frame_end: int,
    params: RenderParams,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    poster_bgr = _read_u8(poster_path)
    h, w = poster_bgr.shape[:2]

    interp = _interp_flag(params.poster_interp)

    for f in range(frame_start, frame_end + 1):
        uv = _read_u16(assets_dir / "uv" / f"uv_{f:04d}.png")
        alpha = _read_u8(assets_dir / "alpha" / f"alpha_{f:04d}.png")[:, :, 0].astype(np.float32) / 255.0
        back = _read_u8(assets_dir / "backside" / f"back_{f:04d}.png")[:, :, 0].astype(np.float32) / 255.0
        light = _read_u16(assets_dir / "light" / f"light_{f:04d}.png")[:, :, 0].astype(np.float32) / 65535.0

        map_x, map_y = decode_uv_to_mapxy(uv, h=h, w=w)
        warped = cv2.remap(poster_bgr, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)

        warped_lin = _srgb_to_linear(warped.astype(np.float32) / 255.0)
        base_lin = warped_lin * (1.0 - back[:, :, None]) + 1.0 * back[:, :, None]

        ratio = light * float(params.light_scale)
        out_lin = base_lin * ratio[:, :, None]
        out_srgb = np.clip(_linear_to_srgb(np.clip(out_lin, 0.0, 8.0)), 0.0, 1.0)

        out_bgr = (out_srgb * 255.0 + 0.5).astype(np.uint8)
        out_bgr = (out_bgr.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)

        cv2.imwrite(str(out_dir / f"frame_{f:04d}.png"), out_bgr)

