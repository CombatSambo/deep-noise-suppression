#!/bin/bash
#SBATCH -J dns_encdec_eval
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
EVAL_DIR=outputs/split_test_eval_encdec
NOISY_DIR="$EVAL_DIR/noisy"
ENHANCED_DIR="$EVAL_DIR/enhanced"

cd "$PROJECT_DIR"

module load singularity

mkdir -p logs "$EVAL_DIR"

singularity exec --nv -B /gpfs:/gpfs "$CONTAINER" bash -lc "
  cd '$PROJECT_DIR'
  source .venv/bin/activate

  echo '===== GPU check ====='
  nvidia-smi

  echo '===== Evaluate noisy test files ====='
  python evaluate_enhanced.py \
    --input '$NOISY_DIR' \
    --metrics sigmos dnsmos \
    --csv-out '$EVAL_DIR/noisy_metrics.csv' \
    --json-out '$EVAL_DIR/noisy_metrics.json'

  echo '===== Evaluate enhanced test files ====='
  python evaluate_enhanced.py \
    --input '$ENHANCED_DIR' \
    --metrics sigmos dnsmos \
    --csv-out '$EVAL_DIR/enhanced_metrics.csv' \
    --json-out '$EVAL_DIR/enhanced_metrics.json'

  echo '===== Evaluation finished ====='
  ls -lh '$EVAL_DIR'/*metrics.*
"
