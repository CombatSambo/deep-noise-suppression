from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch
import yaml

from data_pipeline import build_train_dataloader, build_train_dataset_from_config_dict


def _build_synthetic_config(sample_rate: int, segment_seconds: float, num_examples: int, root_dir: Path) -> Dict[str, Any]:
    clean_dir = root_dir / "clean"
    noise_dir = root_dir / "noise"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    durations = [0.5, 0.9, 1.7]

    for idx, duration in enumerate(durations):
        num_samples = int(round(duration * sample_rate))
        t = np.linspace(0.0, duration, num_samples, endpoint=False)
        tone = 0.2 * np.sin(2.0 * np.pi * (220.0 + 40.0 * idx) * t).astype(np.float32)
        sf.write(str(clean_dir / f"clean_{idx}.wav"), tone, sample_rate)

    for idx, duration in enumerate(durations):
        noise_sr = sample_rate if idx % 2 == 0 else sample_rate // 2
        num_samples = int(round(duration * noise_sr))
        noise = (0.05 * rng.standard_normal(num_samples)).astype(np.float32)
        sf.write(str(noise_dir / f"noise_{idx}.wav"), noise, noise_sr)

    return {
        "onlinesynth_sampling_rate": sample_rate,
        "onlinesynth_resampling_type": "kaiser_fast",
        "onlinesynth_nearend_snr_interval": [-5.0, 20.0],
        "onlinesynth_nearend_normalize_volume": -27,
        "onlinesynth_nearend_datasets": {
            "synthetic_clean": {
                "weight": 1.0,
                "dir": str(clean_dir),
            }
        },
        "onlinesynth_nearend_noises": {
            "synthetic_noise": {
                "weight": 1.0,
                "dir": str(noise_dir),
            }
        },
        "train_data": {
            "segment_seconds": segment_seconds,
            "epoch_size": num_examples,
            "allow_resample": True,
            "clip_protection": True,
            "random_gain_db_range": [-3.0, 3.0],
            "file_extensions": [".wav"],
        },
        "train_dataloader": {
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
            "drop_last": False,
            "pin_memory": False,
        },
    }


def _validate_batch(batch: Dict[str, Any], expected_sample_rate: int) -> None:
    required_keys = {"clean", "noisy", "noise", "snr_db", "sample_rate", "clean_path", "noise_path"}
    missing = required_keys - set(batch.keys())
    if missing:
        raise AssertionError(f"Missing keys in batch: {sorted(missing)}")

    clean = batch["clean"]
    noisy = batch["noisy"]
    noise = batch["noise"]
    snr_db = batch["snr_db"]
    sample_rate = batch["sample_rate"]

    assert clean.ndim == 3, f"Expected clean shape [B, C, T], got {tuple(clean.shape)}"
    assert noisy.shape == clean.shape, f"Noisy shape {tuple(noisy.shape)} must match clean {tuple(clean.shape)}"
    assert noise.shape == clean.shape, f"Noise shape {tuple(noise.shape)} must match clean {tuple(clean.shape)}"
    assert clean.dtype == torch.float32, f"Expected float32 clean tensor, got {clean.dtype}"
    assert noisy.dtype == torch.float32, f"Expected float32 noisy tensor, got {noisy.dtype}"
    assert noise.dtype == torch.float32, f"Expected float32 noise tensor, got {noise.dtype}"
    assert snr_db.ndim == 1, f"Expected snr_db shape [B], got {tuple(snr_db.shape)}"
    assert sample_rate.ndim == 1, f"Expected sample_rate shape [B], got {tuple(sample_rate.shape)}"
    assert int(sample_rate[0].item()) == expected_sample_rate, (
        f"Expected sample rate {expected_sample_rate}, got {int(sample_rate[0].item())}"
    )
    assert np.all(np.isfinite(clean.numpy())), "Clean batch contains non-finite values"
    assert np.all(np.isfinite(noisy.numpy())), "Noisy batch contains non-finite values"
    assert np.all(np.isfinite(noise.numpy())), "Noise batch contains non-finite values"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for DNS training data pipeline (Blocks 1-2).")
    parser.add_argument("--config", type=str, default="synthesizer_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--epoch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train_data", {})
    cfg.setdefault("train_dataloader", {})
    cfg["train_data"]["segment_seconds"] = args.segment_seconds
    cfg["train_data"]["epoch_size"] = args.epoch_size
    cfg["train_data"].setdefault("allow_resample", True)
    cfg["train_data"].setdefault("clip_protection", True)
    cfg["train_data"].setdefault("random_gain_db_range", [-3.0, 3.0])
    cfg["train_data"].setdefault("file_extensions", [".wav"])
    cfg["train_dataloader"]["batch_size"] = args.batch_size
    cfg["train_dataloader"].setdefault("num_workers", 0)
    cfg["train_dataloader"].setdefault("shuffle", True)
    cfg["train_dataloader"].setdefault("drop_last", False)
    cfg["train_dataloader"].setdefault("pin_memory", False)

    using_synthetic_fallback = False
    synthetic_tmp_dir = None
    try:
        dataset = build_train_dataset_from_config_dict(cfg=cfg, seed=args.seed)
    except (FileNotFoundError, ValueError):
        using_synthetic_fallback = True
        synthetic_tmp_dir = tempfile.TemporaryDirectory(prefix="dns_data_pipeline_")
        cfg = _build_synthetic_config(
            sample_rate=16_000,
            segment_seconds=args.segment_seconds,
            num_examples=args.epoch_size,
            root_dir=Path(synthetic_tmp_dir.name),
        )
        dataset = build_train_dataset_from_config_dict(cfg=cfg, seed=args.seed)

    dl_cfg = cfg["train_dataloader"]
    dataloader = build_train_dataloader(
        dataset=dataset,
        batch_size=int(dl_cfg["batch_size"]),
        num_workers=int(dl_cfg["num_workers"]),
        shuffle=bool(dl_cfg["shuffle"]),
        drop_last=bool(dl_cfg["drop_last"]),
        pin_memory=bool(dl_cfg["pin_memory"]),
        seed=args.seed,
    )

    batch = next(iter(dataloader))
    _validate_batch(batch=batch, expected_sample_rate=int(cfg["onlinesynth_sampling_rate"]))

    clean = batch["clean"].numpy()
    noisy = batch["noisy"].numpy()
    noise = batch["noise"].numpy()
    print("Data pipeline smoke test: PASS")
    print(f"fallback_synthetic={using_synthetic_fallback}")
    print(f"clean shape={tuple(batch['clean'].shape)}, dtype={batch['clean'].dtype}")
    print(f"noisy shape={tuple(batch['noisy'].shape)}, dtype={batch['noisy'].dtype}")
    print(f"noise shape={tuple(batch['noise'].shape)}, dtype={batch['noise'].dtype}")
    print(f"sample_rate={int(batch['sample_rate'][0].item())}")
    print(f"snr_db[0]={float(batch['snr_db'][0].item()):.3f}")
    print(
        "ranges "
        f"clean=[{clean.min():.4f}, {clean.max():.4f}] "
        f"noisy=[{noisy.min():.4f}, {noisy.max():.4f}] "
        f"noise=[{noise.min():.4f}, {noise.max():.4f}]"
    )

    if synthetic_tmp_dir is not None:
        synthetic_tmp_dir.cleanup()


if __name__ == "__main__":
    main()
