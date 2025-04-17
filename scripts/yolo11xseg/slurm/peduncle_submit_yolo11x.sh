#!/bin/bash
#SBATCH --job-name=yolo11x-seg_train
#SBATCH --output=yolo11x-seg_train_%j.out
#SBATCH --error=yolo11x-seg_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX6000:1     
#SBATCH --mem=128G
#SBATCH --partition=regular

# Email notifications
#SBATCH --mail-user=aja294@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load required modules
module purge
module load cuda/12.6.0
module load anaconda3/2023.03

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate yolo11-env

# Configure environment variables for segmentation optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Print environment information
echo "Python version: $(python --version)"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No GPU detected, running on CPU"
else
    echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
    nvidia-smi
fi

# Create output directory
OUTPUT_DIR="/workdir/data/grape/grape_pheno/grape_peduncle/models/yolo11x-seg_run_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Set paths
ANNOTATIONS_DIR="/workdir/data/grape/grape_pheno/grape_peduncle/data/annotations/yolo11x-seg"
DATA_YAML="${ANNOTATIONS_DIR}/data.yaml"
MODEL="yolo11x-seg.pt"

# Check for checkpoint
RESUME_OPTION=""
CHECKPOINT="${OUTPUT_DIR}/weights/last.pt"
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint found, resuming training"
    RESUME_OPTION="--resume ${CHECKPOINT}"
fi

# Create a command file - this approach avoids line continuation issues
cat > ${OUTPUT_DIR}/train_command.sh << EOF
python scripts/train_yolo11_peduncle.py --data "${DATA_YAML}" --model "${MODEL}" --output-dir "${OUTPUT_DIR}" --epochs 1000 --batch-size 16 --img-size 640 --workers 12 --device cuda:0 --attention sca --use-repvgg --optimizer AdamW --patience 20 --min-delta 0.001 --amp --save-period 5 --verbose --dropout 0.05 --rect ${RESUME_OPTION}
EOF

# Make it executable and run it
chmod +x ${OUTPUT_DIR}/train_command.sh
echo "Running training command:"
cat ${OUTPUT_DIR}/train_command.sh
${OUTPUT_DIR}/train_command.sh

# The rest of your script for evaluation can remain unchanged
if [ -f "${OUTPUT_DIR}/YOLO11X_SEG_TRAINING_COMPLETED" ]; then
    echo "YOLOv11x-seg training completed successfully"
    cat "${OUTPUT_DIR}/YOLO11X_SEG_TRAINING_COMPLETED"

    # Run evaluation on best model
    BEST_MODEL="${OUTPUT_DIR}/weights/yolo11x-seg_best.pt"
    if [ -f "$BEST_MODEL" ]; then
        TEST_OUTPUT="${OUTPUT_DIR}/evaluation"
        mkdir -p "${TEST_OUTPUT}"

        # Create evaluation command file
        cat > ${OUTPUT_DIR}/eval_command.sh << EOF
python scripts/eval_yolo11_segmentation.py --model "${BEST_MODEL}" --data "${DATA_YAML}" --output-dir "${TEST_OUTPUT}" --device cuda:0 --batch-size 8 --task segment --save-json --save-conf --save-txt --save-mask --iou 0.7 --retina-masks
EOF
        chmod +x ${OUTPUT_DIR}/eval_command.sh
        ${OUTPUT_DIR}/eval_command.sh
    else
        echo "Best model not found at ${BEST_MODEL}"
    fi
elif [ -f "${OUTPUT_DIR}/YOLO11X_SEG_TRAINING_FAILED" ]; then
    echo "YOLOv11x-seg training failed"
    cat "${OUTPUT_DIR}/YOLO11X_SEG_TRAINING_FAILED"
else
    echo "YOLOv11x-seg training status unknown"
fi

# Print job end time
echo "End time: $(date)"

# Deactivate conda environment
conda deactivate
