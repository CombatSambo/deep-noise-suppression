from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from models.baseline_pytorch import BaselineGRUMaskNet


@dataclass(frozen=True)
class AudioFrontendConfig:
    sampling_rate: int = 16_000
    window_length_sec: float = 0.02
    hop_fraction: float = 0.5
    dft_size: int = 320


class PyTorchEnhancer:
    def __init__(
        self,
        model: BaselineGRUMaskNet,
        frontend_cfg: Optional[AudioFrontendConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.frontend_cfg = frontend_cfg or AudioFrontendConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.sampling_rate = int(self.frontend_cfg.sampling_rate)
        self.frame_size = int(self.frontend_cfg.window_length_sec * self.sampling_rate)
        self.hop_size = int(self.frame_size * self.frontend_cfg.hop_fraction)
        self.dft_size = int(self.frontend_cfg.dft_size)

        if self.frame_size <= 0 or self.hop_size <= 0:
            raise ValueError("Invalid frame or hop size derived from frontend config.")
        if self.model.config.output_size != (self.dft_size // 2 + 1):
            raise ValueError(
                f"Model output size {self.model.config.output_size} must match dft_size//2+1 "
                f"({self.dft_size // 2 + 1}) for waveform reconstruction."
            )

        self.window = np.sqrt(np.hanning(self.frame_size + 1)[:-1]).astype(np.float32)

    @staticmethod
    def _logpow(sig: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(sig ** 2, 1e-12))

    def _calc_features(self, xmag_mic: np.ndarray, xmag_far: np.ndarray) -> torch.Tensor:
        feat_mic = self._logpow(xmag_mic)
        feat_far = self._logpow(xmag_far)
        feat = np.concatenate([feat_mic, feat_far]).astype(np.float32)
        feat /= 20.0
        return torch.from_numpy(feat[np.newaxis, np.newaxis, :]).to(self.device)

    def enhance_waveforms(
        self,
        mic_wave: np.ndarray,
        far_wave: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        if far_wave is None:
            far_wave = np.zeros_like(mic_wave)

        min_len = min(len(mic_wave), len(far_wave))
        if min_len <= 0:
            raise ValueError("Empty mic/far input.")
        mic_wave = mic_wave[:min_len]
        far_wave = far_wave[:min_len]

        pad_left = self.hop_size
        mic_wave = np.pad(mic_wave, (pad_left, 0))
        far_wave = np.pad(far_wave, (pad_left, 0))

        num_frames = (len(mic_wave) - self.frame_size) // self.hop_size + 1
        ola_output = np.zeros(self.frame_size + (num_frames - 1) * self.hop_size, dtype=np.float32)

        h01, h02 = self.model.init_hidden(batch_size=1, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for ix_start in range(0, len(mic_wave) - self.frame_size, self.hop_size):
                ix_end = ix_start + self.frame_size

                frame_mic = mic_wave[ix_start:ix_end] * self.window
                cspec_mic = np.fft.rfft(frame_mic, self.dft_size)
                xmag_mic = np.abs(cspec_mic).astype(np.float32)
                xphs_mic = np.ones_like(cspec_mic)
                nonzero = xmag_mic > 0.0
                xphs_mic[nonzero] = cspec_mic[nonzero] / xmag_mic[nonzero]

                frame_far = far_wave[ix_start:ix_end] * self.window
                cspec_far = np.fft.rfft(frame_far, self.dft_size)
                xmag_far = np.abs(cspec_far).astype(np.float32)

                feat = self._calc_features(xmag_mic, xmag_far)
                mask, h01, h02 = self.model(feat, h01, h02)
                mask_np = mask[0, 0].cpu().numpy()

                enhanced_frame = np.fft.irfft(mask_np * xmag_mic * xphs_mic, self.dft_size).astype(np.float32)
                enhanced_frame = enhanced_frame[: self.frame_size] * self.window
                ola_output[ix_start:ix_end] += enhanced_frame

        enhanced = ola_output[pad_left:]
        peak = float(np.max(np.abs(enhanced)))
        if peak > 1.0:
            enhanced = enhanced / peak

        meta = {
            "input_num_samples": int(min_len),
            "output_num_samples": int(len(enhanced)),
            "num_frames": int(num_frames),
            "frame_size": int(self.frame_size),
            "hop_size": int(self.hop_size),
            "dft_size": int(self.dft_size),
        }
        return enhanced.astype(np.float32), meta

    def enhance_file(
        self,
        mic_path: Path,
        far_path: Optional[Path],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        mic, mic_sr = sf.read(str(mic_path), always_2d=False, dtype="float32")
        if mic.ndim > 1:
            mic = np.mean(mic, axis=1, dtype=np.float32)
        mic = np.asarray(mic, dtype=np.float32)
        if mic_sr != self.sampling_rate:
            mic = librosa.resample(mic, orig_sr=mic_sr, target_sr=self.sampling_rate).astype(np.float32)

        far = None
        if far_path is not None and far_path.exists():
            far_raw, far_sr = sf.read(str(far_path), always_2d=False, dtype="float32")
            if far_raw.ndim > 1:
                far_raw = np.mean(far_raw, axis=1, dtype=np.float32)
            far = np.asarray(far_raw, dtype=np.float32)
            if far_sr != self.sampling_rate:
                far = librosa.resample(far, orig_sr=far_sr, target_sr=self.sampling_rate).astype(np.float32)

        return self.enhance_waveforms(mic_wave=mic, far_wave=far)


def _collect_wavs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    all_wavs = sorted(input_path.rglob("*.wav"))
    mic_wavs = [p for p in all_wavs if p.name.endswith("_mic.wav")]
    return mic_wavs if mic_wavs else all_wavs


def _resolve_farend_path(
    mic_path: Path,
    farend_dir: Optional[Path],
    farend_suffix: str,
) -> Optional[Path]:
    if farend_suffix and mic_path.name.endswith("_mic.wav"):
        far_name = mic_path.name.replace("_mic.wav", f"{farend_suffix}.wav")
    else:
        far_name = mic_path.name.replace(".wav", f"{farend_suffix}.wav")

    if farend_dir is not None:
        candidate = farend_dir / far_name
        if candidate.exists():
            return candidate
    candidate = mic_path.with_name(far_name)
    return candidate if candidate.exists() else None


def _derive_output_path(mic_path: Path, input_root: Path, output_dir: Path) -> Path:
    rel = mic_path.relative_to(input_root) if input_root.is_dir() else Path(mic_path.name)
    stem = rel.stem
    if stem.endswith("_mic"):
        stem = stem[: -len("_mic")]
    out_name = f"{stem}_enh.wav"
    return output_dir / rel.parent / out_name


def run_enhancement(
    input_path: Path,
    output_dir: Path,
    model: BaselineGRUMaskNet,
    device: str = "cpu",
    sampling_rate: int = 16_000,
    farend_dir: Optional[Path] = None,
    farend_suffix: str = "_lpb",
) -> List[Path]:
    enhancer = PyTorchEnhancer(
        model=model,
        frontend_cfg=AudioFrontendConfig(sampling_rate=sampling_rate),
        device=device,
    )
    wav_paths = _collect_wavs(input_path)
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in '{input_path}'.")

    output_paths: List[Path] = []
    for mic_path in wav_paths:
        far_path = _resolve_farend_path(mic_path, farend_dir=farend_dir, farend_suffix=farend_suffix)
        enhanced, meta = enhancer.enhance_file(mic_path, far_path)

        out_path = _derive_output_path(mic_path, input_root=input_path, output_dir=output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), enhanced, sampling_rate)

        print(
            f"[enhance] mic='{mic_path.name}' far='{far_path.name if far_path is not None else 'None'}' "
            f"frames={meta['num_frames']} input={meta['input_num_samples']} output={meta['output_num_samples']} "
            f"saved='{out_path}'"
        )
        output_paths.append(out_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch baseline enhancement (Blocks 3-4).")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input wav file or directory.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Directory to save enhanced wav files.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Processing sample rate.")
    parser.add_argument(
        "--farend-dir",
        type=str,
        default=None,
        help="Optional directory containing far-end wav files for feature pairing.",
    )
    parser.add_argument(
        "--farend-suffix",
        type=str,
        default="_lpb",
        help="Suffix (without .wav) used to locate far-end files from mic file names.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional PyTorch checkpoint to load.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic eval setup.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    model = BaselineGRUMaskNet()
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[enhance] loaded checkpoint: {ckpt_path}")
    else:
        print("[enhance] no checkpoint provided; using randomly initialized PyTorch baseline model.")

    run_enhancement(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model=model,
        device=args.device,
        sampling_rate=args.sample_rate,
        farend_dir=Path(args.farend_dir) if args.farend_dir else None,
        farend_suffix=args.farend_suffix,
    )


if __name__ == "__main__":
    main()
