'''
OVERVIEW
Purpose: Train a Mask R-CNN model for instance segmentation.
Dataset: The dataset is loaded from COCO-formatted annotations.
Model: The model is initialized with pre-trained weights and supports
       multiple ResNet backbones (ResNet50, ResNet101, ResNet152) that
       can be selected via command-line arguments.
Training: The model is trained for a specified number of epochs.
Checkpoints: The script supports resuming training from checkpoints,
             saving checkpoints at specified intervals, and creating
             backbone-specific symlinks for better organization.
Evaluation: The model is evaluated on a designated validation set.
Logging: Comprehensive logging and error handling are included.
Framework: The training is performed using the PyTorch framework.
Devices: The script was first tested on metal but now prioritizes CUDA.
         This script can run on various devices (CUDA, MPS, or CPU)
         based on availability.
Mixed Precision: Optional mixed precision training is supported on
                 compatible devices for improved performance.
Author: aja294@cornell.edu

'''

# =============================================
# IMPORTS
# =============================================

# ----------------------------------------------
# # Standard Library Imports
# ----------------------------------------------
import argparse      # Command-line argument parsing
import gc            # Garbage collection for memory management
import os            # File and directory operations
import sys           # System-specific parameters and functions
import time          # Time access and conversions
import traceback     # Stack trace for exception handling
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ----------------------------------------------
# Third-party Libraries
# ----------------------------------------------
import numpy as np   # Numerical processing

# ----------------------------------------------
# PyTorch & Vision Libraries
# ----------------------------------------------
import torch
import torch.nn as nn
import torchvision
from torch.amp import GradScaler, autocast  # Mixed precision training
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models.detection import (  # Model building components
    FasterRCNNPredictor,
    MaskRCNNPredictor,
)
import torchvision.transforms as T  # Image transformations


# =============================================
# ARGUMENT PARSING
# =============================================

def parse_args():
    """
    Parse command line arguments for Mask R-CNN training.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN model for instance segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ----------------------------------------------
    # REQUIRED ARGUMENTS
    # ----------------------------------------------
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path to data directory containing COCO format annotations'
    )
    required.add_argument(
        '--checkpoint-path', 
        type=str, 
        required=True,
        help='Path to save model checkpoints'
    )

    # ----------------------------------------------
    # DATA CONFIGURATION
    # ----------------------------------------------
    data_group = parser.add_argument_group('Data Configuration', 
        'Parameters for dataset location and structure')
    data_group.add_argument(
        '--train-dir', 
        type=str, 
        default='coco/train',
        help='Subdirectory for training data relative to data-path'
    )
    data_group.add_argument(
        '--valid-dir', 
        type=str, 
        default='coco/valid',
        help='Subdirectory for validation data relative to data-path'
    )
    data_group.add_argument(
        '--annotations-filename', 
        type=str, 
        default='_annotations.coco.json',
        help='Filename for COCO annotations in data directories'
    )
    data_group.add_argument(
        '--batch-size-train', 
        type=int, 
        default=4,
        help='Batch size for training'
    )
    data_group.add_argument(
        '--batch-size-val', 
        type=int, 
        default=6,
        help='Batch size for validation'
    )
    data_group.add_argument(
        '--num-workers', 
        type=int, 
        default=4,
        help='Number of worker threads for data loading'
    )

    # ----------------------------------------------
    # MODEL ARCHITECTURE
    # ----------------------------------------------
    model_group = parser.add_argument_group('Model Architecture', 
        'Parameters defining the Mask R-CNN model structure')
    model_group.add_argument(
        '--backbone', 
        type=str, 
        default='resnet50',
        choices=['resnet50', 'resnet101', 'resnet152'],
        help='Backbone architecture for Mask R-CNN'
    )
    model_group.add_argument(
        '--mask-predictor-layers', 
        type=int, 
        default=4,
        help='Number of convolutional layers in mask predictor (4 or more)'
    )
    model_group.add_argument(
        '--hidden-layer-size', 
        type=int, 
        default=256,
        help='Size of hidden layer in mask predictor'
    )

    # ----------------------------------------------
    # TRAINING PARAMETERS
    # ----------------------------------------------
    training_group = parser.add_argument_group('Training Parameters',
        'General settings for the training process')
    training_group.add_argument(
        '--num-epochs', 
        type=int, 
        default=5,
        help='Number of training epochs'
    )
    training_group.add_argument(
        '--resume', 
        action='store_true',
        help='Resume training from latest checkpoint if available'
    )
    training_group.add_argument(
        '--disable-early-stopping', 
        action='store_true',
        help='Disable early stopping and train for the full number of epochs'
    )

    # ----------------------------------------------
    # OPTIMIZATION SETTINGS
    # ----------------------------------------------
    optim_group = parser.add_argument_group('Optimization Settings',
        'Parameters for optimizer and learning rate scheduler')
    optim_group.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.005,
        help='Initial learning rate'
    )
    optim_group.add_argument(
        '--momentum', 
        type=float, 
        default=0.9,
        help='Momentum factor for SGD optimizer'
    )
    optim_group.add_argument(
        '--weight-decay', 
        type=float, 
        default=0.0005,
        help='Weight decay (L2 penalty)'
    )
    optim_group.add_argument(
        '--lr-step-size', 
        type=int, 
        default=3,
        help='Period of learning rate decay (epochs)'
    )
    optim_group.add_argument(
        '--lr-gamma', 
        type=float, 
        default=0.1,
        help='Multiplicative factor of learning rate decay'
    )
    optim_group.add_argument(
        '--grad-clip', 
        type=float, 
        default=None,
        help='Value for gradient clipping (None to disable)'
    )

    # ----------------------------------------------
    # EARLY STOPPING CONFIGURATION
    # ----------------------------------------------
    early_stop_group = parser.add_argument_group('Early Stopping Configuration',
        'Parameters controlling when training should terminate')
    early_stop_group.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=3,
        help='Number of epochs to wait for improvement before stopping'
    )
    early_stop_group.add_argument(
        '--early-stopping-delta', 
        type=float, 
        default=0.001,
        help='Minimum change to qualify as improvement'
    )

    # ----------------------------------------------
    # CHECKPOINT & SAVING CONFIGURATION
    # ----------------------------------------------
    checkpoint_group = parser.add_argument_group('Checkpoint Configuration',
        'Parameters for model saving and checkpointing')
    checkpoint_group.add_argument(
        '--checkpoint-freq', 
        type=int, 
        default=1,
        help='Save checkpoints every N epochs (1 = every epoch)'
    )
    checkpoint_group.add_argument(
        '--checkpoint-format', 
        type=str, 
        default="mask_rcnn_{backbone}_{type}_epoch_{epoch}.pth",
        help='Format string for checkpoint naming (supports {backbone}, {type}, {epoch})'
    )
    checkpoint_group.add_argument(
        '--symlink-name', 
        type=str, 
        default='latest',
        help='Name of symlink to create pointing to the latest checkpoint directory'
    )
    checkpoint_group.add_argument(
        '--disable-symlink', 
        action='store_true',
        help='Disable creation of symlink to latest checkpoint directory'
    )

    # ----------------------------------------------
    # HARDWARE & PERFORMANCE OPTIONS
    # ----------------------------------------------
    hardware_group = parser.add_argument_group('Hardware & Performance',
        'Device and performance-related settings')
    hardware_group.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device to use (cuda, mps, cpu). If not specified, will use best available'
    )
    hardware_group.add_argument(
        '--mixed-precision', 
        action='store_true',
        help='Enable mixed precision training when using CUDA or MPS'
    )

    # ----------------------------------------------
    # LOGGING & DEBUGGING
    # ----------------------------------------------
    logging_group = parser.add_argument_group('Logging & Debugging',
        'Options for logging and debugging')
    logging_group.add_argument(
        '--log-interval', 
        type=int, 
        default=10,
        help='How often to log batch results during training'
    )
    logging_group.add_argument(
        '--detailed-loss', 
        action='store_true',
        help='Log detailed breakdown of loss components'
    )

    return parser.parse_args()


# =============================================
# TOP LEVEL FUNCTIONS FOR MASK-RCNN TRAINING
# =============================================

# ---------------------------------------------
# 1. DATA HANDLING
# ---------------------------------------------

# ---------------------------------------------
# 1a. DATASET CLASS
# ---------------------------------------------

class CocoDataset(torch.utils.data.Dataset):
    """
    Dataset class for COCO-style object detection and segmentation.
    """
    def __init__(self, root: str, annotations_file: str, transforms: Optional[Callable] = None):
        """
        Args:
            root (str): Path to the image directory.
            annotations_file (str): Path to the COCO annotations JSON file.
            transforms (Optional[Callable]): Transforms to apply to images and targets.
        """
        self.root = root
        self.transforms = transforms
        self.coco = torchvision.datasets.CocoDetection(root, annotations_file)
        self.ids = list(sorted(self.coco.ids))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the image and target dictionary.
        """
        img_id = self.ids[index]
        img, target = self.coco[index]

        boxes = []
        labels = []
        masks = []
        area = []
        iscrowd = []

        for annotation in target:
            bbox = annotation['bbox']
            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height
            boxes.append([x1, y1, x2, y2])

            category_id = annotation['category_id']
            labels.append(category_id)

            if 'segmentation' in annotation:
                mask = self.coco.coco.annToMask(annotation)
                masks.append(mask)

            # Calculate area if not provided
            area.append(annotation.get('area', width * height))  # Use get() for default

            # Set iscrowd if not provided
            iscrowd.append(annotation.get('iscrowd', 0))  # Use get() for default

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Handle masks
        if masks:
            masks = np.stack(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            # Create empty mask tensor if no masks are present
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)  # Use img dimensions

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)  # Apply transforms to both image and target

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


# ---------------------------------------------
# 1b. DATA LOADING
# ---------------------------------------------


# Transform
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Collate
def collate_fn(batch):
    return tuple(zip(*batch))


# Load datasets and create data loaders
def load_datasets(args, log_message):
    """
    Load datasets and create data loaders based on command line arguments

    Args:
        args (argparse.Namespace): Command line arguments
        log_message (function): Function to log messages

    Returns:
        tuple: Contains the following elements:
            - train_loader (DataLoader): DataLoader for training data
            - val_loader (DataLoader or None): DataLoader for validation data
            - num_classes (int): Number of classes in the dataset (including background)
    """
    log_message("Loading datasets...")

    # Define directories for train and validation
    train_dir = os.path.join(args.data_path, args.train_dir)
    val_dir = os.path.join(args.data_path, args.valid_dir)

    # Define annotation files
    train_annotations_file = os.path.join(train_dir, args.annotations_filename)
    val_annotations_file = os.path.join(val_dir, args.annotations_filename)

    # Create directories if they don't exist
    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)

    # Training dataset
    train_dataset = LeafDataset(
        train_dir, 
        train_annotations_file, 
        get_transform(train=True),
        log_fn=log_message
    )

    # Validate the dataset
    if not train_dataset.validate():
        raise ValueError("Training dataset validation failed")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    log_message(f"Training dataset loaded with {len(train_dataset)} images")

    # Get number of classes from training dataset
    num_classes = len(train_dataset.coco.coco.getCatIds()) + 1  # +1 for background
    log_message(f"Number of classes: {num_classes}")

    # Validation dataset
    val_dataset = None
    val_loader = None
    if os.path.exists(val_annotations_file):
        val_dataset = LeafDataset(
            val_dir, 
            val_annotations_file, 
            get_transform(train=False),
            log_fn=log_message
        )

        # Only create loader if validation dataset is valid
        if val_dataset.validate():
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size_val,
                shuffle=False,
                num_workers=max(1, args.num_workers // 2),  # Use fewer workers for validation
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=True if len(val_dataset) > 20 else False
            )
            log_message(f"Validation dataset loaded with {len(val_dataset)} images")
        else:
            log_message("WARNING: Validation dataset failed validation checks")
    else:
        log_message("WARNING: Validation dataset not available")

    # Check if the training dataset is not empty
    if len(train_dataset) == 0:
        log_message("ERROR: Training dataset is empty!")
        raise ValueError("Training dataset contains no images")

    return train_loader, val_loader, num_classes


# ---------------------------------------------
# 2. MODEL ARCHITECTURE & BACKBONE CONFIG
# ---------------------------------------------


# Model architecture
def get_instance_segmentation_model(num_classes: int, backbone: str = "resnet50",
                                    pretrained: bool = True, hidden_layer: int = 256,
                                    mask_predictor_layers: int = 4) -> torchvision.models.detection.MaskRCNN:
    """
    Create an instance segmentation model with configurable backbone and parameters

    Args:
        num_classes (int): Number of classes including background
        backbone (str): Backbone network - "resnet50", "resnet101", or "resnet152"
        pretrained (bool): Whether to use pretrained weights
        hidden_layer (int): Number of features in the hidden layer
        mask_predictor_layers (int): Number of conv layers in mask predictor

    Returns:
        model: Configured Mask R-CNN model
    """
    # Select appropriate weights and model based on backbone
    if backbone == "resnet50":
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    elif backbone == "resnet101":
        weights = torchvision.models.detection.MaskRCNN_ResNet101_FPN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet101_fpn(weights=weights)
    elif backbone == "resnet152":
        # For ResNet152, we need to construct it manually as it's not directly provided
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        # Set weights based on pretrained flag - this correctly handles the pretrained parameter
        weights_backbone = torchvision.models.ResNet152_Weights.DEFAULT if pretrained else None

        # Create FPN backbone with ResNet152 and appropriate weights
        backbone_model = resnet_fpn_backbone(
            backbone_name="resnet152", 
            weights=weights_backbone,  
            trainable_layers=5
        )

        # Create Mask R-CNN with custom backbone - use actual num_classes directly
        model = torchvision.models.detection.MaskRCNN(backbone_model, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50', 'resnet101', or 'resnet152'")

    # For ResNet50 and ResNet101, we need to replace the classification head
    if backbone in ["resnet50", "resnet101"]:
        # Replace the pre-trained classification head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor with potentially higher capacity one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Use custom mask predictor with more layers for finer details if requested
    if mask_predictor_layers > 4:
        # Create a custom mask predictor with more layers for better edge detail
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor as DefaultPredictor

        class EnhancedMaskPredictor(DefaultPredictor):
            def __init__(self, in_channels, dim_reduced, num_classes):
                super(EnhancedMaskPredictor, self).__init__(in_channels, dim_reduced, num_classes)
                # Add extra convolutional layers for finer edge detection
                extra_layers = []
                for _ in range(mask_predictor_layers - 4):  # Default has 4 layers
                    extra_layers.append(torch.nn.Conv2d(dim_reduced, dim_reduced, 3, 1, 1))
                    extra_layers.append(torch.nn.BatchNorm2d(dim_reduced))  # Added batch normalization
                    extra_layers.append(torch.nn.ReLU(inplace=True))

                self.extra_layers = torch.nn.Sequential(*extra_layers)

            def forward(self, x):
                x = self.conv5_mask(x)
                x = self.relu(x)
                x = self.extra_layers(x)  # Apply extra layers
                return self.mask_fcn_logits(x)

        model.roi_heads.mask_predictor = EnhancedMaskPredictor(in_features_mask, hidden_layer, num_classes)
    else:
        # Use the standard mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# System setup
def setup_model_and_optimization(num_classes: int, device: torch.device, log_message: Callable, 
                                 base_checkpoint_dir: Optional[str] = None, backbone: str = "resnet50", 
                                 mask_predictor_layers: int = 4, hidden_layer_size: int = 256, 
                                 learning_rate: float = 0.005, momentum: float = 0.9, 
                                 weight_decay: float = 0.0005, lr_step_size: int = 3, 
                                 lr_gamma: float = 0.1) -> Tuple[nn.Module, torch.optim.Optimizer, 
                                                                   torch.optim.lr_scheduler._LRScheduler, 
                                                                   int, float, int, dict, torch.cuda.amp.GradScaler]:
    """
    Initialize model, optimizer, scheduler, and load checkpoint if available

    Args:
        num_classes (int): Number of classes including background
        device (torch.device): Device to use for model (CPU, CUDA, MPS)
        log_message (Callable): Function for logging messages
        base_checkpoint_dir (Optional[str]): Directory containing checkpoints
        backbone (str): Backbone architecture ("resnet50", "resnet101", or "resnet152")
        mask_predictor_layers (int): Number of layers in mask predictor
        hidden_layer_size (int): Size of hidden layers
        learning_rate (float): Initial learning rate
        momentum (float): SGD momentum parameter
        weight_decay (float): Weight decay (L2 regularization)
        lr_step_size (int): Epochs between learning rate reductions
        lr_gamma (float): Multiplicative factor for learning rate reduction

    Returns:
        Tuple containing:
            - model: Initialized Mask R-CNN model
            - optimizer: Configured optimizer
            - lr_scheduler: Learning rate scheduler
            - start_epoch: Epoch to start training from
            - best_val_loss: Best validation loss from previous training
            - early_stopping_counter: Current early stopping counter
            - model_config: Dictionary of model architecture configuration
            - scaler: Gradient scaler for mixed precision training
    """
    # Initialize model
    log_message(f"Initializing Mask R-CNN model with {backbone} backbone...")
    log_message(f"  - Mask predictor: {mask_predictor_layers} layers with hidden size {hidden_layer_size}")

    model = get_instance_segmentation_model(
        num_classes,
        backbone=backbone,
        pretrained=True,
        hidden_layer=hidden_layer_size,
        mask_predictor_layers=mask_predictor_layers
    )
    model.to(device)

    # Count the total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_message(f"Model initialized with {total_params:,} total parameters")
    log_message(f"  - {trainable_params:,} trainable parameters ({trainable_params/total_params:.1%})")
    log_message(f"Model moved to device: {device}")

    # Set default values
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    checkpoint = None

    # Create model configuration dictionary for checkpointing
    model_config = {
        'backbone': backbone,
        'mask_predictor_layers': mask_predictor_layers,
        'hidden_layer_size': hidden_layer_size,
        'num_classes': num_classes
    }

    # Check for existing checkpoints
    if base_checkpoint_dir:
        log_message(f"Looking for {backbone} checkpoints in {base_checkpoint_dir}")

        # First try to find backbone-specific checkpoint
        backbone_pattern = f"mask_rcnn_{backbone}_"
        backbone_checkpoints = []

        if os.path.exists(base_checkpoint_dir):
            # Look for backbone-specific checkpoints first
            backbone_checkpoints = [
                f for f in os.listdir(base_checkpoint_dir) 
                if f.endswith('.pth') and backbone_pattern in f and 'checkpoint' in f
            ]

            # Sort by epoch number if available
            if backbone_checkpoints:
                try:
                    backbone_checkpoints.sort(
                        key=lambda x: int(x.split('_epoch_')[1].split('.')[0]) 
                        if '_epoch_' in x else 0, 
                        reverse=True
                    )
                except Exception:
                    # If sorting fails, just use the original order
                    pass

        latest_checkpoint_path = None

        # If backbone-specific checkpoints found, use the first one
        if backbone_checkpoints:
            latest_checkpoint_path = os.path.join(base_checkpoint_dir, backbone_checkpoints[0])
            log_message(f"Found backbone-specific checkpoint: {os.path.basename(latest_checkpoint_path)}")
        else:
            # Check for best model with this backbone
            best_model_path = os.path.join(base_checkpoint_dir, f"mask_rcnn_{backbone}_best_model.pth")
            if os.path.exists(best_model_path):
                latest_checkpoint_path = best_model_path
                log_message("Found backbone-specific best model checkpoint")
            else:
                # Last resort: check for generic latest checkpoint
                generic_latest = os.path.join(base_checkpoint_dir, "mask_rcnn_latest.pth")
                if os.path.exists(generic_latest):
                    log_message("WARNING: Only found generic latest checkpoint, backbone may not match")
                    latest_checkpoint_path = generic_latest
                else:
                    log_message(f"No checkpoints found for {backbone} backbone")

        # Try to load the checkpoint if one was found
        if latest_checkpoint_path:
            try:
                log_message(f"Backbone-Specific Checkpoints: {backbone_checkpoints}")
                log_message(f"Loading checkpoint: {latest_checkpoint_path}")
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                loaded_config = checkpoint.get('model_config', {})  # Get config from checkpoint

                # Check for configuration mismatches
                mismatches = []

                # Check backbone first - this is critical
                checkpoint_backbone = loaded_config.get('backbone')
                if checkpoint_backbone and checkpoint_backbone != backbone:
                    mismatches.append(f"backbone: {checkpoint_backbone} != {backbone}")

                # Check other parameters
                for k, v in model_config.items():
                    if k != 'backbone' and loaded_config.get(k) is not None and loaded_config.get(k) != v:
                        mismatches.append(f"{k}: {loaded_config.get(k)} != {v}")

                if mismatches:
                    log_message("WARNING: Model architecture mismatch in checkpoint:")
                    for mismatch in mismatches:
                        log_message(f"  - {mismatch}")

                    if 'backbone' in [m.split(':')[0].strip() for m in mismatches]:
                        raise ValueError(f"Cannot load checkpoint with backbone mismatch. Expected {backbone}.")

                    log_message("Continuing with mismatched architecture parameters (may cause issues)")

                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                log_message(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")

            except Exception as e:
                log_message(f"Error loading checkpoint: {e}")
                log_message("Starting training from scratch")
                # Reset values since we failed to load
                start_epoch = 0
                best_val_loss = float('inf')
                early_stopping_counter = 0
                checkpoint = None

    # Set up optimizer and learning rate scheduler
    log_message(f"Setting up optimizer with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    log_message(f"Setting up lr_scheduler with step_size={lr_step_size}, gamma={lr_gamma}")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Load optimizer and scheduler states if resuming
    if checkpoint is not None and start_epoch > 0:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_message("Loaded optimizer and scheduler states")
        except Exception as e:
            log_message(f"Error loading optimizer/scheduler states: {e}")

    log_message("Optimizer and learning rate scheduler initialized")

    # Mixed precision setup
    log_message(f"Setting up mixed precision training (enabled={torch.cuda.is_available() or torch.backends.mps.is_available()})")
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available() or torch.backends.mps.is_available()
    )
    log_message(f"Mixed precision enabled: {scaler.is_enabled()}")

    # Load scaler state if resuming from checkpoint
    if checkpoint is not None and start_epoch > 0 and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            log_message("Loaded GradScaler state from checkpoint")
        except Exception as e:
            log_message(f"Error loading GradScaler state: {e}")

    return model, optimizer, lr_scheduler, start_epoch, best_val_loss, early_stopping_counter, model_config, scaler


# ---------------------------------------------
# 3. TRAINING UTILITIES
# ---------------------------------------------


# Train epoch
def train_epoch(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: torch.utils.data.DataLoader,
                device: torch.device,
                log_message: Callable,
                scaler: Optional[torch.cuda.amp.GradScaler] = None,
                epoch_num: Optional[int] = None,
                total_epochs: Optional[int] = None,
                log_interval: int = 10,
                grad_clip_value: Optional[float] = None,
                detailed_loss: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    Run a single training epoch

    Args:
        model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use
        train_loader (DataLoader): DataLoader for training data
        device (torch.device): Device to run model on
        log_message (function): Function to log messages
        scaler (GradScaler, optional): Gradient scaler for mixed precision training
        epoch_num (int, optional): Current epoch number for logging
        total_epochs (int, optional): Total number of epochs for logging
        log_interval (int, optional): How often to log batch results
        grad_clip_value (float, optional): Value for gradient clipping (None to disable)
        detailed_loss (bool, optional): Whether to log detailed loss components

    Returns:
        tuple: Contains:
            - avg_train_loss (float): Average training loss for the epoch
            - loss_components (dict): Average of each loss component
    """
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    batch_count = 0

    # Track individual loss components
    loss_components = {
        'loss_classifier': 0,
        'loss_box_reg': 0,
        'loss_mask': 0,
        'loss_objectness': 0,
        'loss_rpn_box_reg': 0
    }

    if epoch_num is not None and total_epochs is not None:
        log_message(f"Epoch {epoch_num}/{total_epochs}")

    # Calculate total batches for progress reporting
    total_batches = len(train_loader)

    # Get initial GPU memory usage if available
    gpu_memory_start = None
    if device.type == 'cuda':
        try:
            gpu_memory_start = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        except:
            pass

    for i, (images, targets) in enumerate(train_loader):
        batch_start_time = time.time()

        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients
        optimizer.zero_grad()

        # Use mixed precision if on CUDA and scaler is provided
        if device.type == 'cuda' and scaler is not None:
            # Forward pass with autocast
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward pass with scaling
            scaler.scale(losses).backward()

            # Apply gradient clipping if specified
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular precision training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            # Apply gradient clipping if specified
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()

        # Update running loss totals
        epoch_loss += losses.item()
        batch_count += 1

        # Update individual loss components
        for k, v in loss_dict.items():
            if k in loss_components:
                loss_components[k] += v.item()

        # Log progress periodically
        if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
            batch_time = time.time() - batch_start_time
            progress = (i + 1) / total_batches * 100

            # Basic log message
            log_str = f"  Batch {i+1}/{total_batches} ({progress:.1f}%), "
            log_str += f"Loss: {losses.item():.4f}, "
            log_str += f"Time: {batch_time:.2f}s"

            # Add detailed loss components if requested
            if detailed_loss:
                component_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                if component_str:
                    log_str += f"\n    Components: {component_str}"

            log_message(log_str)

        # Clean up memory
        del images, targets, loss_dict, losses
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Calculate average losses
    avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
    avg_loss_components = {k: v / batch_count for k, v in loss_components.items() if batch_count > 0}

    # Final logging for the epoch (improved)
    if epoch_num is not None:
        epoch_duration = time.time() - epoch_start_time
        log_message(f"Epoch {epoch_num} complete. Avg Loss: {avg_train_loss:.4f}, Time: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)")

        # Log average loss components if detailed logging is enabled
        if detailed_loss:
            component_log = ", ".join([f"{k}: {v:.4f}" for k, v in avg_loss_components.items()])
            log_message(f"  Avg Loss Components: {component_log}")

        # Log GPU memory usage if available
        if device.type == 'cuda' and gpu_memory_start is not None:
            try:
                gpu_memory_end = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                log_message(f"  GPU memory: {gpu_memory_end:.1f} MB (change: {gpu_memory_end - gpu_memory_start:.1f} MB)")
            except:
                pass

    return avg_train_loss, avg_loss_components


# Model eval
def evaluate_model(model, data_loader, device, log_message=print):
    """
    Evaluate the model on a dataset without computing gradients.
    Returns the average loss.
    """
    model.train()  # Temporarily set to train mode to compute losses
    total_loss = 0
    batch_count = 0

    try:
        with torch.no_grad():  # No gradients needed for evaluation
            for i, (images, targets) in enumerate(data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass and compute loss
                loss_dict = model(images, targets)

                # Sum all losses
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                batch_count += 1

                if (i + 1) % 5 == 0:  # Log progress
                    log_message(f"  Eval batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")

                # Clear memory
                del images, targets, loss_dict, losses
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model.eval()  # Set back to eval mode
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    except Exception as e:
        model.eval()  # Ensure model is set back to eval mode even if an error occurs
        log_message(f"Error during evaluation: {e}")
        traceback.print_exc()
        return float('inf')


def early_stopping_check(val_loss, best_val_loss, counter, patience, min_delta, log_message, disable_early_stopping=False):
    """
    Check early stopping criteria and update counter

    Args:
        val_loss (float or None): Current validation loss (None if validation not performed)
        best_val_loss (float): Best validation loss seen so far
        counter (int): Current early stopping counter
        patience (int): Maximum number of epochs to wait for improvement
        min_delta (float): Minimum change to qualify as improvement
        log_message (function): Function to log messages
        disable_early_stopping (bool, optional): If True, always return should_stop=False

    Returns:
        tuple: Contains the following elements:
            - best_val_loss (float): Updated best validation loss
            - counter (int): Updated early stopping counter
            - should_stop (bool): Whether training should stop
            - improved (bool): Whether validation loss improved
    """
    should_stop = False
    improved = False

    # Skip early stopping logic if disabled
    if disable_early_stopping:
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            log_message("  Validation loss improved, but early stopping is disabled.")
        return best_val_loss, counter, False, improved

    # Skip early stopping if no validation loss is available
    if val_loss is None:
        log_message("  Early stopping skipped: No validation loss available")
        return best_val_loss, counter, should_stop, improved

    # Check if we have significant improvement
    if val_loss < best_val_loss - min_delta:
        # Improvement found, reset counter
        improvement_amount = best_val_loss - val_loss
        improvement_percent = (improvement_amount / best_val_loss) * 100
        improved = True
        best_val_loss = val_loss
        counter = 0
        log_message(f"  Validation loss improved by {improvement_amount:.4f} ({improvement_percent:.2f}%) to {val_loss:.4f}")
        log_message(f"  Early stopping counter reset (patience: {patience})")
    else:
        # No significant improvement
        counter += 1
        log_message(f"  No significant improvement. Early stopping counter: {counter}/{patience}")

        if val_loss < best_val_loss:
            # There was an improvement, but not significant enough
            log_message(f"  Minor improvement detected ({best_val_loss - val_loss:.6f}), but below threshold ({min_delta})")
        elif val_loss == best_val_loss:
            log_message(f"  Validation loss unchanged at {val_loss:.4f}")
        else:
            # Performance got worse
            log_message(f"  Validation loss increased by {val_loss - best_val_loss:.4f}")

    # Check if we should stop training
    if counter >= patience:
        should_stop = True
        log_message(f"  Early stopping triggered after {counter} epochs without significant improvement")
        log_message(f"  Best validation loss achieved: {best_val_loss:.4f}")

    return best_val_loss, counter, should_stop, improved


# ---------------------------------------------
# 4. MODEL INITIALIZATION AND SETUP
# ---------------------------------------------


# Define setup paths and logging
def setup_paths_and_logging(args):
    """
    Setup logging infrastructure and checkpoint directories

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        tuple: Contains the following elements:
            - checkpoint_dir (str): Directory where checkpoints will be stored
            - log_message (function): Function to log messages to console and file
            - timestamp (str): Timestamp string for this training run
    """
    # Get current timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create checkpoint directory with timestamp
    checkpoint_dir = os.path.join(args.checkpoint_path, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define training and validation paths (for logging purposes)
    train_dir = os.path.join(args.data_path, args.train_dir)
    val_dir = os.path.join(args.data_path, args.valid_dir)
    train_annotations_file = os.path.join(train_dir, args.annotations_filename)
    val_annotations_file = os.path.join(val_dir, args.annotations_filename)

    # Create directories if they don't exist
    for directory in [train_dir, val_dir, os.path.dirname(checkpoint_dir)]:
        os.makedirs(directory, exist_ok=True)

    # Create log file
    log_file = os.path.join(checkpoint_dir, "training_log.txt")

    # Define log function
    def log_message(message):
        """Write message to log file and print to console"""
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

    # Log initial messages
    log_message(f"=== Training started at {timestamp} ===")
    log_message("Command line arguments:")
    for arg, value in sorted(vars(args).items()):
        log_message(f"  {arg}: {value}")

    # Log important paths
    log_message("\nDirectories:")
    log_message(f"  Data path: {args.data_path}")
    log_message(f"  Train path: {train_dir}")
    log_message(f"  Validation path: {val_dir}")
    log_message(f"  Checkpoint directory: {checkpoint_dir}")

    # Check if annotation files exist and log warnings
    missing_files = []
    for file_path, name in [(train_annotations_file, "Training"),
                            (val_annotations_file, "Validation")]:
        if not os.path.exists(file_path):
            missing_files.append((name, file_path))

    if missing_files:
        log_message("\nWARNING: Missing annotation files:")
        for name, path in missing_files:
            log_message(f"  {name} annotations file not found at {path}")
        log_message("Please ensure all annotation files exist before proceeding.\n")

    # Save the configuration as a JSON file for reproducibility
    config_file = os.path.join(checkpoint_dir, "config.json")
    with open(config_file, 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)
    log_message(f"Configuration saved to {config_file}")

    return checkpoint_dir, log_message, timestamp


# ---------------------------------------------
# 4. CHECKPOINTING
# ---------------------------------------------


def find_latest_checkpoint(checkpoint_dir: str, backbone: str = None) -> Optional[str]:
    """Finds the latest checkpoint file in the given directory for a specific backbone.

    Args:
        checkpoint_dir (str): Directory containing checkpoints
        backbone (str, optional): Specific backbone to search for (e.g., 'resnet50')

    Returns:
        Optional[str]: Path to the latest checkpoint or None if not found
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None

    # If backbone is specified, look for backbone-specific checkpoints
    if backbone:
        pattern = f"mask_rcnn_{backbone}_checkpoint_epoch_"
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.endswith('.pth') and pattern in f]

        if checkpoint_files:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
            return os.path.join(checkpoint_dir, checkpoint_files[0])

    # Fall back to any checkpoint if backbone not specified or none found
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pth') and 'mask_rcnn_' in f and 'checkpoint' in f]

    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   scheduler: _LRScheduler, 
                   checkpoint_dir: str, 
                   epoch: int, 
                   losses: Dict[str, float], 
                   counters: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None, 
                   is_best: bool = False, 
                   save_frequency: int = 1,
                   checkpoint_format: str = "mask_rcnn_{backbone}_{type}_epoch_{epoch}.pth",
                   scaler: Optional[GradScaler] = None) -> Optional[str]:
    """
    Save model checkpoint with configurable naming and frequency.

    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save
        checkpoint_dir (str): Directory to save checkpoint in
        epoch (int): Current epoch number
        losses (dict): Dictionary containing loss values (e.g., {'train': 0.1, 'val': 0.2})
        counters (dict): Dictionary containing counters (e.g., {'early_stopping': 2, 'num_classes': 5})
        metadata (dict, optional): Additional metadata to save
        is_best (bool, optional): Whether this is the best model so far
        save_frequency (int, optional): Save checkpoint every N epochs (1 = every epoch)
        checkpoint_format (str, optional): Format string for checkpoint naming with {backbone}, {type}, and {epoch} placeholders
        scaler (GradScaler, optional): Gradient scaler for mixed precision training

    Returns:
        str: Path to the latest saved checkpoint file. If multiple checkpoints 
             were saved (e.g., regular and best), this will be the path to the 
             most recently saved one. Returns None if saving fails.
    """
    # Determine if we should save a checkpoint this epoch
    save_this_epoch = (epoch % save_frequency == 0) or (epoch == 1) or is_best

    if not save_this_epoch and not is_best:
        return None

    try:
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Extract backbone from metadata for filename formatting
        backbone = "unknown"
        if metadata and 'model_config' in metadata and 'backbone' in metadata['model_config']:
            backbone = metadata['model_config']['backbone']

        # Create the checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'timestamp': datetime.datetime.now().isoformat(),
            'model_config': metadata.get('model_config', {}) if metadata else {}  # Include model_config
        }

        # Save scaler state if available (for mixed precision training)
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Add losses
        if losses:
            checkpoint.update(losses)

        # Add counters
        if counters:
            checkpoint.update(counters)

        # Add additional metadata (excluding model_config which was already added)
        if metadata:
            checkpoint.update({k: v for k, v in metadata.items() if k != 'model_config'})

        # Save paths
        saved_paths = []

        # Regular checkpoint if it's time to save
        if save_this_epoch:
            # Regular checkpoint filename using the format string with backbone
            checkpoint_filename = checkpoint_format.format(
                backbone=backbone,
                type="checkpoint", 
                epoch=epoch
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

            # Save the checkpoint
            torch.save(checkpoint, checkpoint_path)
            saved_paths.append(checkpoint_path)

        # If this is the best model, save it with a backbone-specific name
        if is_best:
            fixed_best_path = os.path.join(checkpoint_dir, f"mask_rcnn_{backbone}_best_model.pth")
            torch.save(checkpoint, fixed_best_path)
            saved_paths.append(fixed_best_path)

        # Always save a latest model that overwrites the previous one (backbone-specific)
        latest_model_path = os.path.join(checkpoint_dir, f"mask_rcnn_{backbone}_latest.pth")
        torch.save(checkpoint, latest_model_path)

        # Also save a generic latest model for backward compatibility
        generic_latest_path = os.path.join(checkpoint_dir, "mask_rcnn_latest.pth")
        torch.save(checkpoint, generic_latest_path)

        # Return the most recently saved path, or the latest path if nothing else was saved
        return saved_paths[-1] if saved_paths else latest_model_path

    except Exception as e:
        import traceback
        print(f"Error saving checkpoint: {e}")
        print(traceback.format_exc())
        return None


def create_symlink(checkpoint_dir, log_message, backbone="resnet50", symlink_name="latest", disable_symlink=False):
    """
    Create symlinks to the latest checkpoint directory, including backbone-specific ones

    Args:
        checkpoint_dir (str): Path to the checkpoint directory to link to
        log_message (function): Function to log messages
        backbone (str, optional): Backbone architecture name for specific symlink (default: "resnet50")
        symlink_name (str, optional): Base name of the symlink (default: "latest")
        disable_symlink (bool, optional): If True, skip symlink creation (default: False)
    """
    # Skip if symlinks are disabled
    if disable_symlink:
        log_message("Symlink creation disabled by configuration")
        return

    # Extract the base directory (without the checkpoints_timestamp part)
    base_dir = os.path.dirname(checkpoint_dir)

    # Define both a generic and a backbone-specific symlink
    generic_link = os.path.join(base_dir, symlink_name)
    backbone_link = os.path.join(base_dir, f"{symlink_name}_{backbone}")
    symlinks = [generic_link, backbone_link]

    for link in symlinks:
        # Check if the symlink already exists and remove it
        if os.path.exists(link):
            try:
                if os.path.islink(link):
                    # Unix-like systems use unlink
                    os.unlink(link)
                else:
                    # It exists but is not a symlink (e.g., a directory)
                    log_message(f"Warning: {link} exists but is not a symlink. Removing it.")
                    if os.path.isdir(link):
                        import shutil
                        shutil.rmtree(link)
                    else:
                        os.remove(link)

                log_message(f"Removed existing symlink: {link}")
            except Exception as e:
                log_message(f"Error removing existing symlink {link}: {e}")
                continue  # Skip to next symlink if we can't remove this one

        # Create the symlink pointing to the current checkpoint directory
        try:
            os.symlink(checkpoint_dir, link)
            log_message(f"Created symlink: {link} -> {checkpoint_dir}")
        except Exception as e:
            log_message(f"Error creating symlink {link}: {e}")


# =============================================
# MAIN FUNCTION - TRAIN MASK-RCNN
# =============================================


def main():
    """
    Main training function for Mask R-CNN model with support for multiple backbone architectures.
    Handles training, validation, checkpointing, and early stopping.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Record the start time for the entire training process
    training_start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Setup logging and checkpoint directory
    checkpoint_dir, log_message, timestamp = setup_paths_and_logging(args)

    try:
        # ---------------------------------------------
        # DEVICE SETUP
        # ---------------------------------------------
        log_message("Setting up compute device...")

        if args.device:
            # User explicitly specified a device
            device_name = args.device.lower()
            if device_name == 'cuda' and not torch.cuda.is_available():
                raise ValueError("CUDA specified but not available.")
            elif device_name == 'mps' and not torch.backends.mps.is_available():
                raise ValueError("MPS specified but not available.")
            device = torch.device(device_name)
        elif torch.cuda.is_available():  # Prioritize CUDA if available
            device = torch.device('cuda')
            log_message(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            log_message("MPS available (Apple Silicon)")
        else:
            device = torch.device('cpu')
            log_message("No GPU acceleration available, using CPU")

        log_message(f"Using device: {device}")

        # Set up mixed precision if requested and supported
        scaler = None
        if args.mixed_precision and (device.type == 'cuda' or device.type == 'mps'):
            try:
                scaler = GradScaler()
                log_message("Mixed precision training enabled")
            except (ImportError, AttributeError) as e:
                log_message(f"Warning: Mixed precision requested but not supported: {e}")
                log_message("Continuing with full precision")

        # ---------------------------------------------
        # DATA LOADING
        # ---------------------------------------------
        log_message("Loading datasets...")
        try:
            train_loader, val_loader, num_classes = load_datasets(args, log_message)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading datasets: {e}. Check that data paths exist.")
        except ValueError as e:
            raise ValueError(f"Dataset validation error: {e}")

        # ---------------------------------------------
        # MODEL SETUP
        # ---------------------------------------------
        log_message(f"Setting up Mask R-CNN model with {args.backbone} backbone...")
        try:
            model, optimizer, lr_scheduler, start_epoch, best_val_loss, early_stopping_counter, model_config, scaler = setup_model_and_optimization(
                num_classes,
                device,
                log_message,
                args.checkpoint_path if args.resume else None,
                backbone=args.backbone,
                mask_predictor_layers=args.mask_predictor_layers,
                hidden_layer_size=args.hidden_layer_size,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma
            )
        except ValueError as e:
            raise ValueError(f"Error setting up model: {e}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(f"CUDA out of memory error. Try reducing batch size or model complexity: {e}")
            else:
                raise RuntimeError(f"Model initialization error: {e}")

        # Training parameters from args
        num_epochs = args.num_epochs
        log_message(f"Starting training for {num_epochs} epochs (resuming from epoch {start_epoch})")
        log_message(f"Model: Mask R-CNN with {args.backbone} backbone, {num_classes} classes")

        # Print dataset info
        log_message(f"Training dataset: {len(train_loader.dataset)} images, {len(train_loader)} batches")
        if val_loader:
            log_message(f"Validation dataset: {len(val_loader.dataset)} images, {len(val_loader)} batches")
        else:
            log_message("Note: No validation dataset provided - early stopping will be disabled")

        # ---------------------------------------------
        # TRAINING LOOP
        # ---------------------------------------------
        log_message("\n" + "="*80)
        log_message(f"TRAINING START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message("="*80)

        # Metrics tracking
        train_losses = []
        val_losses = []

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            log_message(f"\nEpoch {epoch+1}/{num_epochs} - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Run training epoch
            avg_train_loss, loss_components = train_epoch(
                model,
                optimizer,
                train_loader,
                device,
                log_message,
                scaler=scaler,
                epoch_num=epoch+1,
                total_epochs=num_epochs,
                log_interval=args.log_interval,
                grad_clip_value=args.grad_clip,
                detailed_loss=args.detailed_loss
            )
            train_losses.append(avg_train_loss)

            # Validation phase
            val_loss = None
            if val_loader:
                log_message("Running validation...")
                val_loss = evaluate_model(model, val_loader, device, log_message)
                val_losses.append(val_loss)
                log_message(f"  Validation Loss: {val_loss:.4f}")
            else:
                log_message("Skipping validation (no validation data provided)")

            # Update learning rate
            lr_scheduler.step()
            log_message(f"  Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

            # Check early stopping criteria
            best_val_loss, early_stopping_counter, should_stop, improved = early_stopping_check(
                val_loss,
                best_val_loss,
                early_stopping_counter,
                args.early_stopping_patience,
                args.early_stopping_delta,
                log_message,
                disable_early_stopping=args.disable_early_stopping or val_loader is None
            )

            # Prepare losses and counters for checkpoint
            losses = {
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }

            # Add individual loss components if detailed logging is enabled
            if args.detailed_loss:
                for k, v in loss_components.items():
                    losses[f'train_{k}'] = v

            counters = {
                'early_stopping_counter': early_stopping_counter,
                'num_classes': num_classes
            }

            # Save checkpoint with backbone in the format
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                checkpoint_dir,
                epoch + 1,
                losses,
                counters,
                metadata={'model_config': model_config},
                is_best=improved,
                save_frequency=args.checkpoint_freq,
                checkpoint_format=args.checkpoint_format,
                scaler=scaler
            )

            if checkpoint_path:
                log_message(f"  Checkpoint saved to {checkpoint_path}")
            else:
                next_save = ((epoch+1)//args.checkpoint_freq+1)*args.checkpoint_freq
                next_save = min(next_save, num_epochs)
                log_message(f"  No checkpoint saved this epoch (next save at epoch {next_save})")

            # Log epoch duration
            epoch_duration = time.time() - epoch_start_time
            log_message(f"  Epoch completed in {epoch_duration:.2f}s ({epoch_duration/60:.2f} min)")

            # Break training loop if early stopping triggered
            if should_stop:
                log_message("\nEarly stopping triggered - no significant improvement for "
                           f"{args.early_stopping_patience} epochs. Halting training.")
                break

        # ---------------------------------------------
        # FINALIZATION
        # ---------------------------------------------
        # Save the final model with backbone-specific name
        final_model_path = os.path.join(
            checkpoint_dir, f"maskrcnn_{args.backbone}_final.pth"
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'train_loss': avg_train_loss,
            'val_loss': val_loss if val_loader else None,
            'model_config': model_config,
            'timestamp': timestamp,
            'device': str(device),
            'args': vars(args)  # Save all arguments for reproducibility
        }, final_model_path)
        log_message(f"Final model saved to {final_model_path}")

        # Create symlink to latest checkpoint directory, including backbone-specific link
        create_symlink(
            checkpoint_dir, 
            log_message, 
            backbone=args.backbone,
            symlink_name=args.symlink_name, 
            disable_symlink=args.disable_symlink
        )

        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Final message with total training time
        total_time = time.time() - training_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        log_message("\n" + "="*80)
        log_message(f"TRAINING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        log_message(f"Final training loss: {avg_train_loss:.4f}")
        if val_loader:
            log_message(f"Final validation loss: {val_loss:.4f}")
            log_message(f"Best validation loss: {best_val_loss:.4f}")
        log_message("="*80)

        return 0  # Success exit code

    except KeyboardInterrupt:
        log_message("\nTraining interrupted by user")
        return 130  # Standard exit code for SIGINT

    except ValueError as e:
        log_message(f"ERROR: Configuration error: {e}")
        return 1

    except FileNotFoundError as e:
        log_message(f"ERROR: File not found: {e}")
        return 2

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            log_message(f"ERROR: GPU memory insufficient: {e}")
            log_message("Try reducing batch size or model complexity")
            return 3
        else:
            log_message(f"ERROR: Runtime error: {e}")
            log_message(traceback.format_exc())
            return 4

    except Exception as e:
        log_message(f"ERROR: Unexpected error: {e}")
        log_message(traceback.format_exc())
        return 5

    finally:
        # Basic cleanup 
        gc.collect()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
