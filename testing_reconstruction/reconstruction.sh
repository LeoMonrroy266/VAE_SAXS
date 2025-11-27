#!/bin/bash

# Check if there are exactly 2 arguments (model_path and mode)
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_path> <mode> <latent_size>"
    echo "mode should be one of: 1, 2, 3"
    exit 1
fi

# Get the model_path and mode from the arguments
MODEL_PATH=$1
MODE=$2
Z=$3
NAME=$(basename "$MODEL_PATH")

# Map mode number to mode name
if [ "$MODE" -eq 1 ]; then
    MODE_NAME="annealing"
elif [ "$MODE" -eq 2 ]; then
    MODE_NAME="bayesian"
elif [ "$MODE" -eq 3 ]; then
    MODE_NAME="ga"
else
    echo "Invalid mode. Please select mode 1 (annealing), 2 (bayesian), or 3 (ga)."
    exit 1
fi

# Run the python script with the corresponding mode name
python3 "/home/x_leomo/saxsdiff/users/VAE_SAXS/main_reconstruction_${MODE_NAME}.py" \
    --iq_path "/home/x_leomo/saxsdiff/users/VAE_SAXS/testing_reconstruction/test_files/diff_data.dat" \
    --pdb "/home/x_leomo/saxsdiff/users/VAE_SAXS/testing_reconstruction/test_files/Ref_pred.pdb" \
    --output_folder "/home/x_leomo/saxsdiff/users/VAE_SAXS/testing_reconstruction/${MODE_NAME}/${NAME}" \
    --model_path "$MODEL_PATH" \
    --latent_size $Z


