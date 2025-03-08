#!/bin/sh

SOURCE_FILE=$1
DESTINATION_FILE=$2

if [[ -f $SOURCE_FILE && -f $DESTINATION_FILE ]]; then
    tail -n+2 $SOURCE_FILE | cat >> $DESTINATION_FILE
    rm -f $SOURCE_FILE
elif [[ -f $SOURCE_FILE ]]; then
    echo "[append_files.sh] Missing destination file: $DESTINATION_FILE, skipping ..."
elif [[ -f $DESTINATION_FILE ]]; then
    echo "[append_files.sh] Missing $SOURCE_FILE, skip ..."
else
    echo "[append_files.sh] Missing everything!"
fi
