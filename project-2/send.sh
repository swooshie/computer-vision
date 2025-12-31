#!/bin/bash

# --- hpc_upload.sh ---
#
# A simple script to upload files or directories to the NYU HPC cluster.
# It automatically selects the correct destination (/scratch for code, /vast for data)
# based on the first argument.
#
# Usage: ./hpc_upload.sh [code|data] /path/to/your/local/file_or_directory
#

# --- Configuration ---
# Your NetID
NETID="aaj6301"

# The SSH alias for the HPC login node (from your ~/.ssh/config file)
HPC_ALIAS="aaj6301@greene.hpc.nyu.edu"

# Destination paths on the HPC
CODE_DEST_PATH="/scratch/$NETID/cell_segmentation_project"
DATA_DEST_PATH="/vast/$NETID/"
# --- End Configuration ---


# 1. Check if the user provided exactly two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [code|data] /path/to/local/file_or_directory"
    echo "Example: $0 code ./my-project-folder"
    echo "Example: $0 data ~/Downloads/my-dataset.zip"
    exit 1
fi

# 2. Assign arguments to clear variable names
TYPE="$1"
LOCAL_PATH="$2"

# 3. Check if the local file/directory actually exists
if [ ! -e "$LOCAL_PATH" ]; then
    echo "Error: File or directory not found at '$LOCAL_PATH'"
    exit 1
fi

# 4. Determine the destination and run the scp command
case "$TYPE" in
  "code")
    echo "Uploading CODE to '$CODE_DEST_PATH'..."
    # Use scp -r (recursive) to copy entire directories
    scp -r "$LOCAL_PATH" "${HPC_ALIAS}:${CODE_DEST_PATH}"
    ;;
  
  "data")
    echo "Uploading DATA to '$DATA_DEST_PATH'..."
    # Use scp -r (recursive) to copy entire directories
    scp -r "$LOCAL_PATH" "${HPC_ALIAS}:${DATA_DEST_PATH}"
    ;;
  
  *)
    echo "Error: Invalid type '$TYPE'. Must be 'code' or 'data'."
    echo "Usage: $0 [code|data] /path/to/local/file_or_directory"
    exit 1
    ;;
esac

echo ""
echo "Upload complete!"