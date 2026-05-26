from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + eps))


def _normalize_to_db(x: np.ndarray, target_level_db: float, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    scalar = 10.0 ** (target_level_db / 20.0) / (_rms(x, eps=eps) + eps)
    return x * scalar, scalar


def _load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _resample_if_needed(
    audio: np.ndarray,
    src_sr: int,
    dst_sr: int,
    resample_type: str,
    allow_resample: bool,
    path: Path,
) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if not allow_resample:
        raise ValueError(
            f"Sample-rate mismatch for '{path}': file SR={src_sr}, expected SR={dst_sr}. "
            "Enable resampling or provide matched audio."
        )
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr, res_type=resample_type).astype(np.float32)


def _random_crop_or_pad(audio: np.ndarray, target_num_samples: int, rng: np.random.Generator) -> np.ndarray:
    length = int(audio.shape[0])
    if length == target_num_samples:
        return audio
    if length > target_num_samples:
        start = int(rng.integers(0, length - target_num_samples + 1))
        return audio[start : start + target_num_samples]

    out = np.zeros(target_num_samples, dtype=np.float32)
    pad_total = target_num_samples - length
    start = int(rng.integers(0, pad_total + 1))
    out[start : start + length] = audio
    return out


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass(frozen=True)
class _AudioGroup:
    name: str
    files: Tuple[Path, ...]
    weight: float


class _WeightedAudioPool:
    def __init__(
        self,
        datasets_cfg: Mapping[str, Mapping[str, Any]],
        file_extensions: Sequence[str],
        recursive: bool,
    ) -> None:
        groups: List[_AudioGroup] = []
        normalized_exts = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_extensions)

        for dataset_name, data_cfg in datasets_cfg.items():
            weight = float(data_cfg.get("weight", 1.0))
            if weight <= 0:
                continue

            dataset_dir = Path(str(data_cfg["dir"]))
            files: List[Path] = []
            if recursive:
                for ext in normalized_exts:
                    files.extend(dataset_dir.rglob(f"*{ext}"))
            else:
                for ext in normalized_exts:
                    files.extend(dataset_dir.glob(f"*{ext}"))
            files = sorted(p for p in files if p.is_file())

            if not files:
                raise ValueError(f"Dataset '{dataset_name}' has no audio files in '{dataset_dir}'.")
            groups.append(_AudioGroup(name=dataset_name, files=tuple(files), weight=weight))

        if not groups:
            raise ValueError("No non-empty audio datasets were found.")

        self.groups = groups
        weights = np.asarray([g.weight for g in groups], dtype=np.float64)
        self.group_probs = weights / weights.sum()

    def sample_path(self, rng: np.random.Generator) -> Tuple[str, Path]:
        group_idx = int(rng.choice(len(self.groups), p=self.group_probs))
        group = self.groups[group_idx]
        file_idx = int(rng.integers(0, len(group.files)))
        return group.name, group.files[file_idx]

    def total_files(self) -> int:
        return int(sum(len(group.files) for group in self.groups))


class SpeechNoiseDataset(Dataset):
    """
    On-the-fly clean/noisy pair synthesis dataset for speech enhancement training.
    """

    def __init__(
        self,
        clean_datasets_cfg: Mapping[str, Mapping[str, Any]],
        noise_datasets_cfg: Mapping[str, Mapping[str, Any]],
        sample_rate: int,
        segment_seconds: float,
        snr_db_range: Tuple[float, float],
        num_examples: int,
        seed: Optional[int] = None,
        file_extensions: Sequence[str] = (".wav", ".flac"),
        recursive: bool = True,
        allow_resample: bool = True,
        resample_type: str = "kaiser_fast",
        random_gain_db_range: Optional[Tuple[float, float]] = (-6.0, 6.0),
        target_level_db: Optional[float] = None,
        clip_protection: bool = True,
    ) -> None:
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if num_examples <= 0:
            raise ValueError("num_examples must be > 0")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(segment_seconds * sample_rate))
        self.snr_db_min = float(min(snr_db_range))
        self.snr_db_max = float(max(snr_db_range))
        self.num_examples = int(num_examples)
        self.seed = seed
        self.allow_resample = bool(allow_resample)
        self.resample_type = str(resample_type)
        self.random_gain_db_range = random_gain_db_range
        self.target_level_db = target_level_db
        self.clip_protection = bool(clip_protection)

        self.clean_pool = _WeightedAudioPool(
            datasets_cfg=clean_datasets_cfg,
            file_extensions=file_extensions,
            recursive=recursive,
        )
        self.noise_pool = _WeightedAudioPool(
            datasets_cfg=noise_datasets_cfg,
            file_extensions=file_extensions,
            recursive=recursive,
        )

    def __len__(self) -> int:
        return self.num_examples

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        seed_seq = np.random.SeedSequence([int(self.seed), int(index)])
        return np.random.default_rng(seed_seq)

    def _load_and_match_length(self, path: Path, rng: np.random.Generator) -> np.ndarray:
        audio, sr = _load_audio_mono(path)
        audio = _resample_if_needed(
            audio=audio,
            src_sr=sr,
            dst_sr=self.sample_rate,
            resample_type=self.resample_type,
            allow_resample=self.allow_resample,
            path=path,
        )
        return _random_crop_or_pad(audio=audio, target_num_samples=self.segment_samples, rng=rng)

    def _apply_gain(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.random_gain_db_range is None:
            return x
        gain_db = float(rng.uniform(self.random_gain_db_range[0], self.random_gain_db_range[1]))
        return x * _db_to_linear(gain_db)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rng = self._rng_for_index(index)

        clean_dataset, clean_path = self.clean_pool.sample_path(rng)
        noise_dataset, noise_path = self.noise_pool.sample_path(rng)

        clean = self._load_and_match_length(clean_path, rng)
        noise = self._load_and_match_length(noise_path, rng)

        clean = self._apply_gain(clean, rng)
        noise = self._apply_gain(noise, rng)

        snr_db = float(rng.uniform(self.snr_db_min, self.snr_db_max))
        clean_rms = _rms(clean)
        noise_rms = _rms(noise)
        noise_scale = clean_rms / (noise_rms + 1e-12) / _db_to_linear(snr_db)
        noise_scaled = noise * noise_scale
        noisy = clean + noise_scaled

        if self.target_level_db is not None:
            noisy, scalar = _normalize_to_db(noisy, target_level_db=float(self.target_level_db))
            clean = clean * scalar
            noise_scaled = noise_scaled * scalar

        if self.clip_protection:
            peak = float(
                max(
                    np.max(np.abs(noisy)),
                    np.max(np.abs(clean)),
                    np.max(np.abs(noise_scaled)),
                )
            )
            if peak > 1.0:
                inv_peak = 1.0 / peak
                noisy = noisy * inv_peak
                clean = clean * inv_peak
                noise_scaled = noise_scaled * inv_peak

        return {
            "clean": torch.from_numpy(clean).unsqueeze(0).to(torch.float32),
            "noisy": torch.from_numpy(noisy).unsqueeze(0).to(torch.float32),
            "noise": torch.from_numpy(noise_scaled).unsqueeze(0).to(torch.float32),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "sample_rate": torch.tensor(self.sample_rate, dtype=torch.int32),
            "clean_path": str(clean_path),
            "noise_path": str(noise_path),
            "clean_dataset": clean_dataset,
            "noise_dataset": noise_dataset,
        }


def _parse_range(cfg: Mapping[str, Any], key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    value = cfg.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Config key '{key}' must be a 2-value range.")


def build_train_dataset_from_config_dict(
    cfg: Mapping[str, Any],
    seed: Optional[int] = None,
) -> SpeechNoiseDataset:
    train_data_cfg = cfg.get("train_data", {})

    sample_rate = int(cfg.get("onlinesynth_sampling_rate", train_data_cfg.get("sample_rate", 16000)))
    segment_seconds = float(
        train_data_cfg.get(
            "segment_seconds",
            cfg.get("onlinesynth_duration", 3.0),
        )
    )
    snr_db_range = _parse_range(cfg, "onlinesynth_nearend_snr_interval", (-5.0, 20.0))
    num_examples = int(train_data_cfg.get("epoch_size", train_data_cfg.get("num_examples", 1000)))
    resample_type = str(cfg.get("onlinesynth_resampling_type", "kaiser_fast"))
    random_gain_db_range = train_data_cfg.get("random_gain_db_range", (-6.0, 6.0))
    if random_gain_db_range is not None:
        random_gain_db_range = (
            float(random_gain_db_range[0]),
            float(random_gain_db_range[1]),
        )

    file_extensions = tuple(train_data_cfg.get("file_extensions", [".wav", ".flac"]))

    return SpeechNoiseDataset(
        clean_datasets_cfg=cfg["onlinesynth_nearend_datasets"],
        noise_datasets_cfg=cfg["onlinesynth_nearend_noises"],
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_db_range=snr_db_range,
        num_examples=num_examples,
        seed=seed,
        file_extensions=file_extensions,
        recursive=bool(train_data_cfg.get("recursive_scan", True)),
        allow_resample=bool(train_data_cfg.get("allow_resample", True)),
        resample_type=resample_type,
        random_gain_db_range=random_gain_db_range,
        target_level_db=cfg.get("onlinesynth_nearend_normalize_volume"),
        clip_protection=bool(train_data_cfg.get("clip_protection", True)),
    )


def build_train_dataset_from_config_path(config_path: str, seed: Optional[int] = None) -> SpeechNoiseDataset:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return build_train_dataset_from_config_dict(cfg, seed=seed)


def build_train_dataloader(
    dataset: SpeechNoiseDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )


def build_train_dataloader_from_config_path(config_path: str, seed: Optional[int] = None) -> DataLoader:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = build_train_dataset_from_config_dict(cfg=cfg, seed=seed)
    dl_cfg = cfg.get("train_dataloader", {})
    return build_train_dataloader(
        dataset=dataset,
        batch_size=int(dl_cfg.get("batch_size", 4)),
        num_workers=int(dl_cfg.get("num_workers", 0)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        drop_last=bool(dl_cfg.get("drop_last", False)),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        seed=seed,
    )
