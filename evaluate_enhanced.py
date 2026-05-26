from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf


def _collect_wavs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.wav"))


def _safe_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if x.ndim > 1:
        x = np.mean(x, axis=1, dtype=np.float32)
    return np.asarray(x, dtype=np.float32), int(sr)


def _evaluate_sigmos(
    files: List[Path],
    model_dir: Path,
) -> List[Dict[str, Any]]:
    try:
        from sigmos import SigMOS
    except Exception as exc:
        raise RuntimeError("SigMOS dependencies are unavailable in the current environment.") from exc

    model_file = model_dir / "model-sigmos_1697718653_41d092e8-epo-200.onnx"
    if not model_file.exists():
        raise FileNotFoundError(f"SigMOS model file not found: {model_file}")

    estimator = SigMOS(model_dir=str(model_dir))
    rows: List[Dict[str, Any]] = []
    for wav_path in files:
        audio, sr = _safe_audio_mono(wav_path)
        scores = estimator.run(audio, sr=sr)
        row = {"file": str(wav_path)}
        for key, value in scores.items():
            row[f"sigmos_{key}"] = float(value)
        rows.append(row)
    return rows


def _evaluate_dnsmos(
    files: List[Path],
    model_dir: Path,
    sample_rate: int = 16_000,
) -> List[Dict[str, Any]]:
    try:
        from dnsmos_local import ComputeScore
    except Exception as exc:
        raise RuntimeError("DNSMOS dependencies are unavailable in the current environment.") from exc

    primary = model_dir / "sig_bak_ovr.onnx"
    p808 = model_dir / "model_v8.onnx"
    if not primary.exists() or not p808.exists():
        raise FileNotFoundError(
            "DNSMOS model files not found. Expected 'sig_bak_ovr.onnx' and 'model_v8.onnx' in "
            f"{model_dir}"
        )

    compute = ComputeScore(str(primary), str(p808))
    rows: List[Dict[str, Any]] = []
    for wav_path in files:
        scores = compute(str(wav_path), sampling_rate=sample_rate)
        row = {"file": str(wav_path)}
        keep = ["OVRL", "SIG", "BAK", "P808_MOS", "OVRL_raw", "SIG_raw", "BAK_raw"]
        for key in keep:
            if key in scores:
                row[f"dnsmos_{key}"] = float(scores[key])
        rows.append(row)
    return rows


def evaluate_files(
    input_path: Path,
    run_sigmos: bool = True,
    run_dnsmos: bool = True,
    sigmos_model_dir: Path = Path("SIGMOS"),
    dnsmos_model_dir: Path = Path("DNSMOS"),
    dnsmos_sample_rate: int = 16_000,
) -> List[Dict[str, Any]]:
    files = _collect_wavs(input_path)
    if not files:
        raise FileNotFoundError(f"No .wav files found in '{input_path}'.")

    merged: Dict[str, Dict[str, Any]] = {str(p): {"file": str(p)} for p in files}
    metric_success = 0

    if run_sigmos:
        try:
            sig_rows = _evaluate_sigmos(files, model_dir=sigmos_model_dir)
            for row in sig_rows:
                merged[row["file"]].update(row)
            metric_success += 1
            print("[evaluate] SigMOS: completed")
        except Exception as exc:
            print(f"[evaluate][warning] SigMOS failed: {exc}")

    if run_dnsmos:
        try:
            dns_rows = _evaluate_dnsmos(files, model_dir=dnsmos_model_dir, sample_rate=dnsmos_sample_rate)
            for row in dns_rows:
                merged[row["file"]].update(row)
            metric_success += 1
            print("[evaluate] DNSMOS: completed")
        except Exception as exc:
            print(f"[evaluate][warning] DNSMOS failed: {exc}")

    if metric_success == 0:
        raise RuntimeError("No evaluation metrics could be computed.")

    return [merged[str(p)] for p in files]


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print(f"[evaluate] evaluated files: {len(rows)}")
    metric_keys = sorted({k for row in rows for k in row.keys() if k != "file"})
    for row in rows:
        parts = [f"file={Path(row['file']).name}"]
        for key in metric_keys:
            if key in row:
                parts.append(f"{key}={row[key]:.4f}")
        print("[evaluate] " + " | ".join(parts))

    numeric_means: Dict[str, float] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            numeric_means[key] = float(np.mean(vals))
    if numeric_means:
        print("[evaluate] mean scores:")
        for key in sorted(numeric_means.keys()):
            print(f"[evaluate]   {key}: {numeric_means[key]:.4f}")


def _write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[evaluate] wrote csv: {csv_path}")


def _write_json(rows: List[Dict[str, Any]], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[evaluate] wrote json: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate enhanced wav files using SigMOS and/or DNSMOS.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input wav file or directory.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["sigmos", "dnsmos"],
        help="Any subset of: sigmos dnsmos",
    )
    parser.add_argument("--sigmos-model-dir", type=str, default="SIGMOS")
    parser.add_argument("--dnsmos-model-dir", type=str, default="DNSMOS")
    parser.add_argument("--dnsmos-sample-rate", type=int, default=16_000)
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    selected = {m.lower() for m in args.metrics}
    rows = evaluate_files(
        input_path=Path(args.input),
        run_sigmos="sigmos" in selected,
        run_dnsmos="dnsmos" in selected,
        sigmos_model_dir=Path(args.sigmos_model_dir),
        dnsmos_model_dir=Path(args.dnsmos_model_dir),
        dnsmos_sample_rate=args.dnsmos_sample_rate,
    )
    _print_summary(rows)

    if args.csv_out:
        _write_csv(rows, Path(args.csv_out))
    if args.json_out:
        _write_json(rows, Path(args.json_out))


if __name__ == "__main__":
    main()
