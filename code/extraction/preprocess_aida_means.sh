#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

function ensure_dir {
    if [ "${1: -1}" != "/" ]; then
        echo "${1}/"
    else
        echo $1
    fi
}

if [ "$#" -lt 1 ]; then
    echo "Not enough arguments."
    echo "Usage: $0 DATA_DIR"
    exit
fi


DATA_DIR=$1
DATA_DIR="$(ensure_dir $DATA_DIR)"
echo "Downloading aida_means.tsv into ${DATA_DIR}."

if [ -f "${DATA_DIR}aida_means.tsv" ]
then
    echo "already downloaded aida_means.tsv"
else
    wget -O ${DATA_DIR}aida_means.tsv.bz2 http://resources.mpi-inf.mpg.de/yago-naga/aida/download/aida_means.tsv.bz2
    sudo bzip2 -d -v4 ${DATA_DIR}aida_means.tsv.bz2
fi

