#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT=$SCRIPT_DIR"/hephaestus/run_preparations.py"

sleep 5s 
/path/to/env $PYTHON_SCRIPT \
    --operation make_all \
    --type d nd \
    --feature-type gdd2 \
    --clean All

echo "Success!"
