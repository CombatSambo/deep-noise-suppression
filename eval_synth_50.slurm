#!/bin/bash
#SBATCH -J dns_eval50
#SBATCH -A ealloc_ati-1-neur
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
CONTAINER=/gpfs/helios/projects/neuralnet_course/track02/containers/pytorch_26.04-py3.sif

cd "$PROJECT_DIR"
module load singularity
mkdir -p logs outputs/synth_eval_50

singularity exec --nv -B /gpfs:/gpfs "$CONTAINER" bash -lc '
  cd /gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
  source .venv/bin/activate

  echo "===== Generate eval set ====="
  python generate_eval_set.py \
    --config synthesizer_config.yaml \
    --output-dir outputs/synth_eval_50 \
    --num-samples 50

  echo "===== Enhance noisy files ====="
  python enhance_pytorch.py \
    --input outputs/synth_eval_50/noisy \
    --output-dir outputs/synth_eval_50/enhanced \
    --checkpoint checkpoints_v100_10ep/baseline_pytorch_last.pt \
    --device cuda

  echo "===== Evaluate noisy input ====="
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  python evaluate_enhanced.py \
    --input outputs/synth_eval_50/noisy \
    --metrics sigmos dnsmos \
    --csv-out outputs/synth_eval_50/eval_noisy.csv \
    --json-out outputs/synth_eval_50/eval_noisy.json

  echo "===== Evaluate enhanced output ====="
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  python evaluate_enhanced.py \
    --input outputs/synth_eval_50/enhanced \
    --metrics sigmos dnsmos \
    --csv-out outputs/synth_eval_50/eval_enhanced.csv \
    --json-out outputs/synth_eval_50/eval_enhanced.json

  echo "===== Done ====="
  ls -lh outputs/synth_eval_50
'