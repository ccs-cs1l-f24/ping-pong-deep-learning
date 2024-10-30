#!/bin/bash

# Get the directory where the script lives
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Define paths relative to the script location
TRAIN_PATH="$(dirname "$SCRIPT_DIR")/dataset/images/train"
VAL_PATH="$(dirname "$SCRIPT_DIR")/dataset/images/val"
CONFIG_PATH="$(dirname "$SCRIPT_DIR")/dataset/opentt.yaml"

# Print the paths for verification
echo "Train Path: $TRAIN_PATH"
echo "Validation Path: $VAL_PATH"
echo "Config Path: $CONFIG_PATH"

# Number of classes
NC=1

# Class names
NAMES=("ball")

# Output details
echo "Number of classes: $NC"
echo "Class names: ${NAMES[@]}"

# Write the configuration to opentt.yaml
cat <<EOL > "$CONFIG_PATH"
# OpenTT Configuration File

train: $TRAIN_PATH
val: $VAL_PATH

# Number of classes (adjust according to your dataset)
nc: $NC

# Class names
names:
  0: ${NAMES[0]}
EOL

# Confirm file creation
echo "Configuration file created at $CONFIG_PATH"
