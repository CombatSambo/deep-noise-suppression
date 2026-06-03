#!/bin/bash
#SBATCH -J dns_encdec_enh
#SBATCH -A ealloc_ati-1-neur
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
CONTAINER=/gpfs/helios/projects/neuralnet_course/track02/containers/pytorch_26.04-py3.sif
CHECKPOINT=checkpoints_encdec_20ep_10s/baseline_pytorch_last.pt
INPUT_DIR=outputs/split_test_eval_encdec/noisy
OUTPUT_DIR=outputs/split_test_eval_encdec/enhanced

cd "$PROJECT_DIR"

module load singularity

mkdir -p logs "$OUTPUT_DIR"

singularity exec --nv -B /gpfs:/gpfs "$CONTAINER" bash -lc "
  cd '$PROJECT_DIR'
  source .venv/bin/activate

  echo '===== GPU check ====='
  nvidia-smi

  echo '===== Enhancement start ====='
  python enhance_pytorch.py \
    --input '$INPUT_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --checkpoint '$CHECKPOINT' \
    --device cuda

  echo '===== Enhancement finished ====='
  find '$OUTPUT_DIR' -maxdepth 1 -type f -name '*.wav' | wc -l
"
