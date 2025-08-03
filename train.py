EPOCHS = 20  # Increased for better convergence
MOSAIC = 0.6  # Better balance for augmentation
MIXUP = 0.1   # Add mixup for better generalization
COPY_PASTE = 0.1  # Help with blended backgrounds
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937  # Standard momentum for AdamW
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False

# New augmentation parameters for shadow/lighting robustness
HSV_H = 0.015  # Hue augmentation
HSV_S = 0.7    # Saturation augmentation  
HSV_V = 0.4    # Value/brightness augmentation
DEGREES = 15.0  # Rotation augmentation
TRANSLATE = 0.1 # Translation augmentation
SCALE = 0.9     # Scale augmentation
SHEAR = 0.0     # Shear augmentation
PERSPECTIVE = 0.0 # Perspective augmentation
FLIPUD = 0.0    # Vertical flip
FLIPLR = 0.5    # Horizontal flip

import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Basic training parameters
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    
    # Augmentation parameters
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=MIXUP, help='Mixup augmentation')
    parser.add_argument('--copy_paste', type=float, default=COPY_PASTE, help='Copy paste augmentation')
    
    # Color/lighting augmentation for shadow robustness
    parser.add_argument('--hsv_h', type=float, default=HSV_H, help='HSV Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=HSV_S, help='HSV Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=HSV_V, help='HSV Value augmentation')
    
    # Geometric augmentation
    parser.add_argument('--degrees', type=float, default=DEGREES, help='Rotation degrees')
    parser.add_argument('--translate', type=float, default=TRANSLATE, help='Translation fraction')
    parser.add_argument('--scale', type=float, default=SCALE, help='Scale fraction')
    parser.add_argument('--shear', type=float, default=SHEAR, help='Shear degrees')
    parser.add_argument('--perspective', type=float, default=PERSPECTIVE, help='Perspective fraction')
    parser.add_argument('--flipud', type=float, default=FLIPUD, help='Vertical flip probability')
    parser.add_argument('--fliplr', type=float, default=FLIPLR, help='Horizontal flip probability')
    
    args = parser.parse_args()
    
    this_dir = r'C:/Users/techb/Downloads/Hackathon_Dataset/HackByte_Dataset'
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    
    print("Training with enhanced augmentation for better generalization...")
    print(f"Epochs: {args.epochs}")
    print(f"Mosaic: {args.mosaic}, Mixup: {args.mixup}, Copy-paste: {args.copy_paste}")
    print(f"HSV augmentation - H: {args.hsv_h}, S: {args.hsv_s}, V: {args.hsv_v}")
    
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device= 0,  # Consider using GPU if available
        single_cls=args.single_cls, 
        
        # Core augmentation
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        
        # Color/lighting augmentation (crucial for shadows)
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s, 
        hsv_v=args.hsv_v,
        
        # Geometric augmentationpython 
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        
        # Optimizer settings
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        
        # Additional settings for better generalization
        patience=5,  # Early stopping patience
        save_period=5,  # Save checkpoint every 5 epochs
        val=True,  # Enable validation during training
    )
    
    print("Training completed!")
    print("Check validation metrics during training to monitor overfitting.")

