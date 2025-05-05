#!/bin/bash

# Step 1: Run inference on full images
sh test.sh

# Step 2: Crop images and process annotations
python cropimage.py

# Step 3: Run inference on cropped images
sh testcrop.sh

# Step 4: Restore bounding boxes
python restoredbbox.py

# Step 5: Fusion and evaluation
python fusion.py