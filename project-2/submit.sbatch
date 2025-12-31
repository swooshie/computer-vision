#!/bin/bash
#SBATCH --job-name=cell_seg_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output_%j.log
#SBATCH --account=pr_130_general

# --- Automated Cleanup Step (as requested) ---
echo "Cleaning up previous package installations from home directory..."
rm -rf /home/aaj6301/.local/lib/python3.10
rm -rf /home/aaj6301/.local/bin
rm -rf /home/aaj6301/.cache/pip
rm -rf /home/aaj6301/.conda
echo "Cleanup complete."

# --- Navigate to Project Directory ---
cd /scratch/aaj6301/cell_segmentation_project

# --- Define Container ---
CONTAINER_IMAGE=./pytorch_latest.sif

echo "Starting job in Singularity container..."

# --- Execute Workflow Inside Container ---
singularity exec --nv \
    --bind /scratch/aaj6301:/scratch/aaj6301 \
    --bind /vast/aaj6301:/vast/aaj6301 \
    $CONTAINER_IMAGE \
    bash -c "
        # 1. Create a Conda environment from the YAML file.
        #    This installs everything needed, including system libraries for OpenCV.
        echo 'Creating Conda environment from environment.yml...'
        conda env create -f environment.yml
        
        # 2. Activate the newly created environment.
        source activate cellseg_env
        
        # 3. Run the main Python script.
        echo 'Starting Python training script...'
        python main.py --epochs 30 --batch_size 8 --base_dir /vast/aaj6301/kaggle-data
        
        # 4. Deactivate and remove the environment to save space.
        echo 'Script finished. Cleaning up Conda environment...'
        conda deactivate
        conda env remove -n cellseg_env
    "

echo "Job completed."