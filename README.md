# Enhanced YOLO Object Detection

A robust object detection system built on YOLOv8 with advanced data augmentation techniques for improved performance in challenging lighting conditions and complex backgrounds.

## 🎯 Project Overview

This project addresses critical limitations in traditional object detection systems by implementing enhanced data augmentation techniques specifically designed for:
- **Shadow-resilient detection**
- **Complex background handling**
- **Lighting variation robustness**
- **Industrial safety equipment detection**

## 🏆 Key Features

- **Advanced Augmentation Pipeline**: Mosaic, Mixup, Copy-Paste, and HSV augmentation
- **Shadow Robustness**: HSV color space manipulation for lighting variations
- **Real-time Performance**: Maintains YOLO's speed advantages (6.6ms inference)
- **High Accuracy**: Achieves 85.5% mAP50 on test dataset
- **Production Ready**: Early stopping, checkpointing, and validation monitoring

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **Anaconda**: Latest version recommended
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for models and datasets

## 🚀 Quick Start

### 1. Install Anaconda
Download and install Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download)

### 2. Create and Activate Environment
bash
# Create conda environment
conda create -n edu python=3.8

# Activate environment
conda activate edu


### 3. Clone and Setup Project
bash
# Clone the repository
git clone https://github.com/Surya-PS03/code-catalysts


### 4. Run Environment Setup (Windows)
bash
# For Windows users
setup_env.bat


### 5. Install Dependencies
bash
pip install ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy

### Note: Due to the large size of the dataset and trained model files, they are not included in the GitHub repository. Please ensure correct paths are set in your scripts and double-check for any path-related anomalies when running the code locally.

### 6. Download Dataset

# Dataset URL: https://storage.googleapis.com/duality-public-share/Datasets/Hackathon_Dataset.zip


**Expected Dataset Structure After Download:**

```
Hackathon_Dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── yolo_params.yaml
└── yolov8s.pt
```




### 7. Train the Model
bash
# Run training with the downloaded dataset
python train.py


### 8. Run Predictions
bash
# After training is complete, run predictions
python predict.py


## ⚙️ Configuration Parameters

### Core Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 20 | Number of training epochs |
| optimizer | AdamW | Optimization algorithm |
| lr0 | 0.001 | Initial learning rate |
| lrf | 0.0001 | Final learning rate |

### Augmentation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| mosaic | 0.6 | Mosaic augmentation probability |
| mixup | 0.1 | Mixup augmentation probability |
| copy_paste | 0.1 | Copy-paste augmentation probability |
| hsv_h | 0.015 | Hue augmentation range |
| hsv_s | 0.7 | Saturation augmentation range |
| hsv_v | 0.4 | Brightness augmentation range |

### Geometric Augmentation
| Parameter | Default | Description |
|-----------|---------|-------------|
| degrees | 15.0 | Rotation range (degrees) |
| translate | 0.1 | Translation fraction |
| scale | 0.9 | Scale variation range |
| fliplr | 0.5 | Horizontal flip probability |


---

## 📊 Expected Results

After successful training & prediction, you should see:

| Class            | Images | Instances | Box(P) | R     | mAP50 | mAP50-95 |
| ---------------- | ------ | --------- | ------ | ----- | ----- | -------- |
| all              | 400    | 560       | 0.892  | 0.791 | 0.855 | 0.609    |
| FireExtinguisher | 183    | 183       | 0.850  | 0.825 | 0.865 | 0.608    |
| ToolBox          | 193    | 193       | 0.916  | 0.788 | 0.872 | 0.675    |
| OxygenTank       | 184    | 184       | 0.912  | 0.761 | 0.829 | 0.545    |

---


Speed: 0.4ms preprocess, 6.6ms inference, 0.0ms loss, 3.4ms postprocess per image


## 🔧 Complete Setup Workflow

### Step-by-Step Process

1. **Environment Setup:**
   
bash
   # Install Anaconda (if not already installed)
   # Download from: https://www.anaconda.com/download
   
   # Create environment
   conda create -n edu python=3.8
   
   # Activate environment
   conda activate edu


2. **Project Setup:**
   
bash
   # Clone repository
   git clone https://github.com/Surya-PS03/code-catalysts
   cd enhanced-yolo-detection
   
   # Run setup script (Windows)
   setup_env.bat


3. **Install Dependencies:**
   
bash
   pip install ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy


4. **Download and Train:**
   
bash
   # Dataset downloads automatically from:
   # https://storage.googleapis.com/duality-public-share/Datasets/Hackathon_Dataset.zip
   
   # Train the model
   python train.py


5. **Run Predictions:**
   
bash
   # Generate predictions on test data
   python predict.py


## 📁 Output Files

After training, check these locations:

runs/detect/train/
├── weights/
│   ├── best.pt      # Best model weights
│   └── last.pt      # Latest checkpoint
├── results.png      # Training curves
├── confusion_matrix.png
└── val_batch0_pred.jpg  # Validation predictions


## 🚨 Troubleshooting

### Common Issues

**1. Environment Not Activated:**
bash
# Always ensure conda environment is active
conda activate edu
# You should see (edu) in your terminal prompt


**2. CUDA Out of Memory:**
python
# Reduce batch size in train.py
batch_size = 8  # Reduce from default 16


**3. Dataset Download Issues:**
bash
# If automatic download fails, manually download:
# https://storage.googleapis.com/duality-public-share/Datasets/Hackathon_Dataset.zip
# Extract to project directory


**4. Missing Dependencies:**
bash
# Reinstall all dependencies
conda activate edu
pip install --upgrade ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy


### Windows-Specific Setup

**setup_env.bat should contain:**
batch
@echo off
echo Setting up environment for Enhanced YOLO Detection...
conda activate edu
echo Environment activated successfully!
echo Installing required packages...
pip install ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy
echo Setup complete! Ready to run train.py
pause


## 📈 Model Evaluation

### Validate Your Model
python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run validation
results = model.val(data='yolo_params.yaml')
print(f"mAP50: {results.box.map50}")


### Inference on New Images
python
# Load model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
results[0].show()  # Display results


## 🔄 File Structure After Setup

enhanced-yolo-detection/
├── train.py              # Main training script
├── predict.py            # Prediction script
├── setup_env.bat         # Windows environment setup
├── Hackathon_Dataset/    # Downloaded dataset
│   ├── train/
│   ├── val/
│   ├── test/
│   └── yolo_params.yaml
├── runs/                 # Training outputs
└── README.md


## 🎯 Usage Summary

bash
# 1. Setup (one-time)
conda create -n edu python=3.8
conda activate edu
setup_env.bat
pip install ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy

# 2. Run (every time)
conda activate edu
python train.py
python predict.py


## 📝 Dataset Information

- **URL**: https://storage.googleapis.com/duality-public-share/Datasets/Hackathon_Dataset.zip
- **Classes**: FireExtinguisher, ToolBox, OxygenTank
- **Format**: YOLO format with bounding box annotations
- **Size**: ~500MB compressed


---

**Quick Start Commands:**
bash
conda create -n edu python=3.8
conda activate edu
git clone https://github.com/Surya-PS03/code-catalysts
cd enhanced-yolo-detection
setup_env.bat
pip install ultralytics torch torchvision opencv-python matplotlib seaborn pandas numpy
python train.py
python predict.py
