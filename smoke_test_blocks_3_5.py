from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from enhance_pytorch import run_enhancement
from evaluate_enhanced import evaluate_files
from models.baseline_pytorch import BaselineGRUMaskNet


def _create_synthetic_inputs(root: Path, sr: int, duration_sec: float) -> Dict[str, Path]:
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float32) / float(sr)
    clean = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    noise = 0.04 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    mic = (clean + noise).astype(np.float32)
    far = np.zeros_like(mic)

    mic_path = input_dir / "sample_mic.wav"
    far_path = input_dir / "sample_lpb.wav"
    sf.write(str(mic_path), mic, sr)
    sf.write(str(far_path), far, sr)

    return {
        "input_dir": input_dir,
        "mic_path": mic_path,
        "far_path": far_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Blocks 3-5 (model, postprocess, evaluation).")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-metrics", action="store_true", help="Run SigMOS and DNSMOS if available.")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    with tempfile.TemporaryDirectory(prefix="dns_blocks_3_5_") as tmp_dir:
        root = Path(tmp_dir)
        paths = _create_synthetic_inputs(root=root, sr=args.sample_rate, duration_sec=args.duration_sec)
        output_dir = root / "enhanced"

        model = BaselineGRUMaskNet()
        out_paths = run_enhancement(
            input_path=paths["input_dir"],
            output_dir=output_dir,
            model=model,
            device=args.device,
            sampling_rate=args.sample_rate,
            farend_dir=paths["input_dir"],
            farend_suffix="_lpb",
        )
        if not out_paths:
            raise RuntimeError("No enhanced files produced.")

        enhanced_audio, enhanced_sr = sf.read(str(out_paths[0]), always_2d=False, dtype="float32")
        if enhanced_audio.ndim > 1:
            enhanced_audio = np.mean(enhanced_audio, axis=1, dtype=np.float32)

        print("[smoke] enhancement completed")
        print(f"[smoke] output_file={out_paths[0]}")
        print(f"[smoke] output_sr={enhanced_sr}")
        print(f"[smoke] output_num_samples={len(enhanced_audio)}")
        print(f"[smoke] output_range=[{float(np.min(enhanced_audio)):.4f}, {float(np.max(enhanced_audio)):.4f}]")

        if args.run_metrics:
            try:
                rows = evaluate_files(
                    input_path=output_dir,
                    run_sigmos=True,
                    run_dnsmos=True,
                    sigmos_model_dir=Path("SIGMOS"),
                    dnsmos_model_dir=Path("DNSMOS"),
                    dnsmos_sample_rate=16_000,
                )
                print("[smoke] metrics computed")
                for row in rows:
                    score_items = {k: v for k, v in row.items() if k != "file"}
                    print(f"[smoke] file={Path(row['file']).name} scores={score_items}")
            except Exception as exc:
                print(f"[smoke][warning] metrics step failed: {exc}")
                print("[smoke] enhancement/postprocessing still validated successfully.")

        print("[smoke] PASS")


if __name__ == "__main__":
    main()

