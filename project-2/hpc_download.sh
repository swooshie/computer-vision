#!/bin/bash

# --- hpc_download.sh ---
#
# A simple script to download files or directories FROM the NYU HPC cluster
# to your local machine. It assumes paths are relative to /scratch or /vast.
#
# Usage: ./hpc_download.sh [code|data] <remote_path> [local_destination]
#

# --- Configuration ---
# Your NetID
NETID="aaj6301"

# The full SSH address for the HPC login node
HPC_HOST="aaj6301@greene.hpc.nyu.edu"

# Base paths on the HPC
REMOTE_CODE_PATH="/scratch/$NETID/"
REMOTE_DATA_PATH="/vast/$NETID/"
# --- End Configuration ---


# 1. Check if the user provided at least two arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 [code|data] <remote_path> [local_destination]"
    echo "Example (file): $0 code my_project/results.txt ."
    echo "  (Downloads /scratch/aaj6301/my_project/results.txt to your current folder)"
    echo "Example (dir):  $0 data my_dataset_folder ./"
    echo "  (Downloads the entire /vast/aaj6301/my_dataset_folder to your current folder)"
    exit 1
fi

# 2. Assign arguments to clear variable names
TYPE="$1"
REMOTE_RELATIVE_PATH="$2"
LOCAL_DEST_PATH="$3"

# 3. If no local destination is given, default to the current directory (.)
if [ -z "$LOCAL_DEST_PATH" ]; then
    LOCAL_DEST_PATH="."
    echo "No local destination specified. Downloading to current directory."
fi

# 4. Determine the full remote path and run the scp command
case "$TYPE" in
  "code")
    FULL_REMOTE_PATH="${REMOTE_CODE_PATH}${REMOTE_RELATIVE_PATH}"
    echo "Downloading from CODE path: ${HPC_HOST}:${FULL_REMOTE_PATH}"
    # Use scp -r (recursive) to copy entire directories
    scp -r "${HPC_HOST}:${FULL_REMOTE_PATH}" "$LOCAL_DEST_PATH"
    ;;
  
  "data")
    FULL_REMOTE_PATH="${REMOTE_DATA_PATH}${REMOTE_RELATIVE_PATH}"
    echo "Downloading from DATA path: ${HPC_HOST}:${FULL_REMOTE_PATH}"
    # Use scp -r (recursive) to copy entire directories
    scp -r "${HPC_HOST}:${FULL_REMOTE_PATH}" "$LOCAL_DEST_PATH"
    ;;
  
  *)
    echo "Error: Invalid type '$TYPE'. Must be 'code' or 'data'."
    echo "Usage: $0 [code|data] <remote_path> [local_destination]"
    exit 1
    ;;
esac

echo ""
echo "Download complete!"