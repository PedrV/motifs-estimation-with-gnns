#!/bin/sh

NETWORK_FILE_PATH=$1
RANDS="$(($4 + 0))"
RESULT_FILE_PATH=$5
SUBGRAPH_SIZE="$(($3 + 0))"

# Write here the absolute path to the gtrieScannner, is safer ...
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $(dirname "$SCRIPT_DIR"))"/_excluded/gtrieScanner_src_01/"

echo $BASE_DIR
if [[ "directed" == $2 ]]; then
   cd $BASE_DIR && ./gtrieScanner -s $SUBGRAPH_SIZE -m gtrie "dir"$SUBGRAPH_SIZE".gt" -d -g $NETWORK_FILE_PATH -rs 42 -r $RANDS -o $RESULT_FILE_PATH -raw
elif [[ "undirected" == $2 ]]; then
   cd $BASE_DIR && ./gtrieScanner -s $SUBGRAPH_SIZE -m gtrie "undir"$SUBGRAPH_SIZE".gt" -g $NETWORK_FILE_PATH -rs 42 -r $RANDS -o $RESULT_FILE_PATH -raw
fi
