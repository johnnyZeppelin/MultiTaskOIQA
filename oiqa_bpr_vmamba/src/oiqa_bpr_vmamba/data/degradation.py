from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random
import cv2
import numpy as np
from PIL import Image


@dataclass
class DegradationConfig:
    jpeg_quality_range: tuple[int, int] = (20, 90)
    gaussian_sigma_range: tuple[float, float] = (0.0, 0.012)
    blur_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    blur_sigma_range: tuple[float, float] = (0.1, 2.0)
    downsample_scales: tuple[float, ...] = (0.5, 0.75)
    camera_noise_std_range: tuple[float, float] = (0.0, 0.02)
    random_order: bool = True
    seed: int = 3407

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> 'DegradationConfig':
        return cls(
            jpeg_quality_range=tuple(cfg.get('jpeg_quality_range', [20, 90])),
            gaussian_sigma_range=tuple(cfg.get('gaussian_sigma_range', [0.0, 0.012])),
            blur_kernel_sizes=tuple(cfg.get('blur_kernel_sizes', [3, 5, 7])),
            blur_sigma_range=tuple(cfg.get('blur_sigma_range', [0.1, 2.0])),
            downsample_scales=tuple(cfg.get('downsample_scales', [0.5, 0.75])),
            camera_noise_std_range=tuple(cfg.get('camera_noise_std_range', [0.0, 0.02])),
            random_order=bool(cfg.get('random_order', True)),
            seed=int(cfg.get('seed', 3407)),
        )


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def add_gaussian_noise(arr: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    noise = rng.normalvariate
    out = arr.astype(np.float32) / 255.0
    noise_arr = np.empty_like(out, dtype=np.float32)
    it = np.nditer(noise_arr[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        noise_arr[idx + (0,)] = noise(0.0, sigma)
        noise_arr[idx + (1,)] = noise(0.0, sigma)
        noise_arr[idx + (2,)] = noise(0.0, sigma)
        it.iternext()
    out = np.clip(out + noise_arr, 0.0, 1.0)
    return out * 255.0


def add_camera_sensor_noise(arr: np.ndarray, std: float, rng: random.Random) -> np.ndarray:
    out = arr.astype(np.float32) / 255.0
    shot = np.sqrt(np.maximum(out, 1e-6)) * std
    gaussian = np.random.default_rng(rng.randint(0, 10**9)).normal(0.0, shot, size=out.shape)
    out = np.clip(out + gaussian, 0.0, 1.0)
    return out * 255.0


def add_blur(arr: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def add_downsample(arr: np.ndarray, scale: float, rng: random.Random) -> np.ndarray:
    h, w = arr.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    interps = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
    interp = interps[rng.randint(0, len(interps) - 1)]
    low = cv2.resize(arr, (nw, nh), interpolation=interp)
    return cv2.resize(low, (w, h), interpolation=interp)


def add_jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode('.jpg', arr.astype(np.uint8), encode_param)
    if not ok:
        raise RuntimeError('JPEG encoding failed during degradation synthesis.')
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec.astype(np.float32)


def synthesize_random_degradation(img: Image.Image, cfg: DegradationConfig, seed: int | None = None) -> Image.Image:
    rng = random.Random(cfg.seed if seed is None else seed)
    arr = pil_to_bgr(img).astype(np.float32)
    ops = [
        ('blur', lambda x: add_blur(x, rng.choice(cfg.blur_kernel_sizes), rng.uniform(*cfg.blur_sigma_range))),
        ('downsample', lambda x: add_downsample(x, rng.choice(cfg.downsample_scales), rng)),
        ('gaussian_noise', lambda x: add_gaussian_noise(x, rng.uniform(*cfg.gaussian_sigma_range), rng)),
        ('jpeg', lambda x: add_jpeg(x, rng.randint(*cfg.jpeg_quality_range))),
        ('camera_noise', lambda x: add_camera_sensor_noise(x, rng.uniform(*cfg.camera_noise_std_range), rng)),
    ]
    if cfg.random_order:
        rng.shuffle(ops)
    num_ops = rng.randint(2, 4)
    selected = ops[:num_ops]
    out = arr
    for _, fn in selected:
        out = fn(out)
    return bgr_to_pil(out)
