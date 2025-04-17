
# Grape Cluster Peduncle Measurement Pipeline

A machine learning-based image analysis pipeline using Mask R-CNN and YOLO to extract morphological traits from grape cluster images.

## Project Overview

This repository contains a specialized pipeline for automated analysis of grape cluster morphology. The system processes field-collected grape clusters to extract quantitative data on key traits including peduncle length and rachis length with millimeter precision.

## Goals

- Implement Mask R-CNN and YOLO for automated extraction of grape cluster morphological traits
- Process current and future grape cluster images from the Geneva location
- Analyze approximately 150 genotypes with 10-15 images per genotype
- Create a reproducible workflow for ongoing grape cluster analysis

## Features

The pipeline extracts the following grape cluster traits:
- Peduncle length (from insertion point on shoot to first ramification)
- Shoulder point
- Measurements calibrated to millimeter precision
- Integration with QR code identification system

## Scope

### Included
- Processing of standardized grape cluster images
- Python scripts implementing Mask R-CNN and YOLO
- Trait-specific measurement algorithms
- Integration with existing field data collection workflow

### Excluded
- Collection of samples (performed by USDA team)
- Hardware modifications to existing equipment
- Analysis of traits beyond peduncle and shoulder

## Deliverables

### Technical Components
- Complete Python pipeline with Mask R-CNN and YOLO implementation
- Trait measurement scripts
- Configuration files for reproducible analysis
- Integration with existing QR code system

### Documentation
- Technical architecture documentation
- User guide and setup instructions
- Best practices for image collection

### Results
- Quantitative trait datasets
- Statistical validation of measurements
- Workflow for analyzing both current and historical image data

## Getting Started

[Installation and setup instructions will be added here]

## Requirements

- Python 3.8+
- Required packages (to be listed in requirements.txt)
- Samsung Galaxy S24 for image capture
- FieldBook app and QR code system

## Project Status

Pipeline development initiated December 13, 2024. This project supports USDA grape morphological analysis and is coordinated by Arlyn John Ackerman. Client contact is Fred Gouker (USDA). Data collection begins December 16, 2024.

## License

[License information will be added here]


