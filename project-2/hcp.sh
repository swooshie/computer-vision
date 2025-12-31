#!/bin/bash

# --- hpc_ssh.sh ---
#
# A simple script to SSH into the NYU HPC cluster.
# It connects to either the login node or the 'greene-compute' alias.
#
# Usage: ./hpc_ssh.sh [login|compute]
#
# --- IMPORTANT ---
# Before running './hpc_ssh.sh compute', you MUST:
#   1. SSH to the login node (e.g., using './hpc_ssh.sh login')
#   2. Start an 'srun' job.
#   3. Find your compute node name (e.g., 'ga013') using 'squeue'.
#   4. Manually update your ~/.ssh/config file's 'greene-compute' block
#      to set 'HostName' to your new compute node name.
#

# --- Configuration ---
# Your NetID
NETID="aaj6301"

# The full hostname for the login node
LOGIN_HOST="greene.hpc.nyu.edu"

# The alias for the compute node (from your ~/.ssh/config file)
COMPUTE_ALIAS="greene-compute"
# --- End Configuration ---


# 1. Check if the user provided exactly one argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [login|compute]"
    echo "Example: $0 login       (Connects to the login node)"
    echo "Example: $0 compute     (Connects to the compute node defined in ~/.ssh/config)"
    exit 1
fi

TYPE="$1"

# 2. Determine the destination and run the ssh command
case "$TYPE" in
  "login")
    echo "Connecting to login node: ${NETID}@${LOGIN_HOST}..."
    # 'exec' replaces the script process with the ssh process.
    # When you exit ssh, you'll be back at your original terminal.
    exec ssh "${NETID}@${LOGIN_HOST}"
    ;;
  
  "compute")
    echo "Connecting to compute node alias: ${COMPUTE_ALIAS}..."
    echo "Make sure you have updated the HostName in your ~/.ssh/config!"
    sleep 2
    # 'exec' replaces the script process with the ssh process.
    exec ssh "$COMPUTE_ALIAS"
    ;;
  
  *)
    echo "Error: Invalid type '$TYPE'. Must be 'login' or 'compute'."
    echo "Usage: $0 [login|compute]"
    exit 1
    ;;
esac

# This part will only be reached if 'exec' fails
echo "SSH command failed to execute."
