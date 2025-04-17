# Grape Peduncle Measurement Pipeline

This repository contains code for analyzing grape peduncle and rachis measurements using Mask R-CNN and YOLO machine learning techniques. All data, models, and configurations are securely stored on the Cornell BioHPC server.

## Repository Structure

The project on csubi2 is organized as follows:

```
/workdir/data/grape/grape_pheno/grape_peduncle/
├── configs/                  # Configuration files
│   ├── ultralytics-env.yaml  # Conda environment for YOLO with Ultralytics
│   └── yolo11-env.yaml       # Conda environment for YOLOv11
│
├── data/                     # Data storage
│   ├── annotations/          # Annotation files for training models
│   ├── processed/            # Processed measurement data
│   └── raw/                  # Raw grape cluster images
│
├── models/                   # Trained models
│   ├── maskrcnn/             # Mask R-CNN model files
│   │   ├── run_4-16/         # Mask R-CNN training run from April 16
│   │   └── run1/             # Initial Mask R-CNN training run
│   └── yolo11xSeg/           # YOLOv11 segmentation models
│       ├── yolo11_run_*/     # Numbered YOLO training runs (2343-2355)
│       ├── yolo11x-seg_run_*/ # Numbered YOLOv11-seg runs (2358-2362)
│       └── yolo11x-seg.pt    # Best performing model weights
│
└── scripts/                  # Analysis and utility scripts
    ├── maskrcnn/             # Mask R-CNN related scripts
    │   ├── slurm/            # SLURM job submission scripts
    │   │   └── peduncle_submit.sh
    │   └── train_validate/
    │       └── train_mask-rcnn.py
    └── yolo11xSeg/           # YOLO related scripts
        ├── slurm/            # SLURM job submission scripts
        │   └── peduncle_submit_yolo11x.sh
        └── train_validate/
            ├── train_peduncle_mask-rcnn.py
            └── train_yolo11_peduncle.py
```

## Data Storage

All data for this project is stored on the Cornell BioHPC server and is not included in the GitHub repository. The data is located at:

```
/workdir/data/grape/grape_pheno/grape_peduncle/
```

### Data Access

To access the data, you need an account on the Cornell BioHPC server. There are several methods to access the data:

#### 1. Direct Server Access

```bash
# SSH into the server
ssh username@cbsubi2.biohpc.cornell.edu

# Navigate to the project directory
cd /workdir/data/grape/grape_pheno/grape_peduncle/
```

#### 2. SSHFS Mount

You can mount the remote directory to your local machine:

```bash
# Create a mount point
mkdir -p ~/server_mounts/grape_peduncle

# Mount the remote directory
sshfs username@cbsubi2.biohpc.cornell.edu:/workdir/data/grape/grape_pheno/grape_peduncle ~/server_mounts/grape_peduncle
```

#### 3. SFTP Access

For transferring files:

```bash
sftp username@cbsubi2.biohpc.cornell.edu:/workdir/data/grape/grape_pheno/grape_peduncle
```

## Data Organization

### Raw Data

The raw data consists of grape cluster images stored in:
```
/workdir/data/grape/grape_pheno/grape_peduncle/data/raw/
```

### Annotations

Annotation files used for training the models are stored in:
```
/workdir/data/grape/grape_pheno/grape_peduncle/data/annotations/
```

### Processed Data

Measurements and analysis results are stored in:
```
/workdir/data/grape/grape_pheno/grape_peduncle/data/processed/
```

## Models

### Mask R-CNN Models

Mask R-CNN models are stored in:
```
/workdir/data/grape/grape_pheno/grape_peduncle/models/maskrcnn/
```

Training runs:
- ```run1```: Initial training run
- ```run_4-16```: Training run from April 16, 2025

### YOLO Models

YOLO models are stored in:
```
/workdir/data/grape/grape_pheno/grape_peduncle/models/yolo11xSeg/
```

The directory contains multiple training runs:
- YOLOv11 runs: ```yolo11_run_2343``` through ```yolo11_run_2355```
- YOLOv11-seg runs: ```yolo11x-seg_run_2358``` through ```yolo11x-seg_run_2362```
- Best model: ```yolo11x-seg.pt``` (pre-trained weights)

## Environment Setup

To set up the required environment, use one of the provided conda environment files:

```bash
# SSH into the server
ssh username@cbsubi2.biohpc.cornell.edu

# Navigate to the project directory
cd /workdir/data/grape/grape_pheno/grape_peduncle/

# For YOLO with Ultralytics
conda env create -f configs/ultralytics-env.yaml

# OR for YOLOv11
conda env create -f configs/yolo11-env.yaml

# Activate the environment
conda activate yolo11-env  # or ultralytics-env
```

## Typical Workflow

### 1. Data Preparation

```bash
# Organize images in the raw directory
cd /workdir/data/grape/grape_pheno/grape_peduncle/data/raw

# Create annotations (if needed)
# Annotation tools: CVAT, LabelMe, or Roboflow
```

### 2. Model Training

#### Using Mask R-CNN:

```bash
# Submit SLURM job
cd /workdir/data/grape/grape_pheno/grape_peduncle/scripts/maskrcnn/slurm
sbatch peduncle_submit.sh

# Or run directly
cd /workdir/data/grape/grape_pheno/grape_peduncle/scripts/maskrcnn/train_validate
python train_mask-rcnn.py --data_path ../../data/annotations --output_dir ../../models/maskrcnn/new_run
```

#### Using YOLO:

```bash
# Submit SLURM job
cd /workdir/data/grape/grape_pheno/grape_peduncle/scripts/yolo11xSeg/slurm
sbatch peduncle_submit_yolo11x.sh

# Or run directly
cd /workdir/data/grape/grape_pheno/grape_peduncle/scripts/yolo11xSeg/train_validate
python train_yolo11_peduncle.py --data_path ../../data/annotations --output_dir ../../models/yolo11xSeg/new_run
```

### 3. Inference and Measurement

```python
# Example code for inference (to be added to your scripts)
from ultralytics import YOLO

# Load the best model
model = YOLO('/workdir/data/grape/grape_pheno/grape_peduncle/models/yolo11xSeg/yolo11x-seg.pt')

# Run inference on new images
results = model('/workdir/data/grape/grape_pheno/grape_peduncle/data/raw/new_image.jpg')

# Process results to extract peduncle and rachis measurements
# (Implementation details specific to your measurement approach)
```

## Model Versioning and Selection

### Current Best Models

- **Mask R-CNN**: Check the most recent run directory in ```/models/maskrcnn/``` for latest weights
- **YOLO**: ```yolo11x-seg.pt``` is the current best performing model

### Selecting a Model

When choosing which model to use:

1. YOLOv11-seg models (runs 2358-2362) generally offer the best balance of accuracy and speed
2. Mask R-CNN may provide more precise segmentation for challenging cases
3. Check the training logs in each run directory for performance metrics

## Data Backup

The data is backed up weekly to the Cornell University backup system. The backup includes:

1. Raw image data
2. Annotations
3. Model weights
4. Processed results

**Important**: Always work with a copy of the data when performing destructive operations.

To create a personal backup:

```bash
# From your local machine
rsync -avz --progress username@cbsubi2.biohpc.cornell.edu:/workdir/data/grape/grape_pheno/grape_peduncle/data/ ~/local_backup/grape_peduncle/
```

## Running the Code

To run the code in this repository, you'll need to:

1. Clone this GitHub repository
2. Access the data on the Cornell BioHPC server
3. Update the data paths in the configuration files to point to your data location

Example:

```bash
# Clone the repository
git clone https://github.com/username/grape-peduncle-measurement.git
cd grape-peduncle-measurement

# Set up data paths
export DATA_ROOT=/path/to/mounted/data
# OR
export DATA_ROOT=/workdir/data/grape/grape_pheno/grape_peduncle/

# Run a script
python scripts/yolo11xSeg/train_validate/train_yolo11_peduncle.py --config configs/model_config.yaml
```

## Contributing

When contributing to this repository, please note that the data should remain on the server and not be committed to GitHub. Update paths in your code to reference the server location.

### Adding New Models

When adding new training runs:

1. Create a new directory with a descriptive name and date
2. Include a README.md within the run directory documenting:
   - Training parameters
   - Dataset version used
   - Performance metrics
   - Any special considerations

## Contact

For questions about this project or access to the data, please contact the project maintainers.

---

**Note**: This README provides an overview of the project structure and data organization. For detailed usage instructions, please refer to the documentation in the specific script files.

