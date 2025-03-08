#!/bin/bash

###########################################################################################
# Some libraries like Numpy may be installed by the requirements.txt and reinstalled 
# by the "torch-stuff". This is fine!
###########################################################################################

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/venv"
  exit 1
fi

VENV_PATH="$1"

if [ ! -d "$VENV_PATH" ]; then
  echo "Error: $VENV_PATH is not a valid directory."
  exit 1
fi

VENV_PATH_EXE=$VENV_PATH"/bin/python"

TORCH_VERSION=2.1.0
CUDA_VERSION=cu121

BASE_REQUIREMENTS=requirements.txt

$VENV_PATH_EXE -m pip install -r $BASE_REQUIREMENTS
$VENV_PATH_EXE -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/$CUDA_VERSION
$VENV_PATH_EXE -m pip install torch_geometric==2.4.0
$VENV_PATH_EXE -m pip install pyg_lib==0.4.0 -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
$VENV_PATH_EXE -m pip install torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
$VENV_PATH_EXE -m pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
$VENV_PATH_EXE -m pip install torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
$VENV_PATH_EXE -m pip install torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
