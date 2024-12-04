#!/bin/bash
#SBATCH -Jiclbench             # Job name
#SBATCH -N1 --ntasks-per-node=4          # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=12G --gres=gpu:V100:4
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH -t4:00:00

cd ~/Neural-Network-Ninjas/llmft/scripts
source misc/pace_module.sh
source /home/hice1/kboparai3/scratch/cs7643_l/bin/activate

make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-125m NUM_SHOTS=2
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-350m NUM_SHOTS=2
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-1.3b NUM_SHOTS=2
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-125m NUM_SHOTS=16
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-350m NUM_SHOTS=16
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-1.3b NUM_SHOTS=16
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-125m NUM_SHOTS=32
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-350m NUM_SHOTS=32
make in_context PACE=1 NUM_GPU=4 MODEL_NAME=facebook/opt-1.3b NUM_SHOTS=32
