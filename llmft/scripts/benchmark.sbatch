#!/bin/bash
#SBATCH -Jefficientllmbench             # Job name
#SBATCH -N1 --ntasks-per-node=4          # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=48G --gres=gpu:H100:2
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH -t6:00:00

cd ~/Neural-Network-Ninjas/llmft/scripts
source misc/pace_module.sh
source ~/scratch/cs7643_l/bin/activate

# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=128 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=128 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=128 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=16 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=16 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=16 DATA_SET=cola
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b TRAIN_SAMPLE=16 DATA_SET=cola SAVE_MODEL=1
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=16 DATA_SET=qqp
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=16 DATA_SET=qqp
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=16 DATA_SET=qqp

# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=128 
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=128 
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=128 
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=256 BSZ=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=256 BSZ=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=256 BSZ=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=16
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-2.7b TRAIN_SAMPLE=16 SAVE_MODEL=1
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b TRAIN_SAMPLE=16 SAVE_MODEL=1
# make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b TRAIN_SAMPLE=128 SAVE_MODEL=1

make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b
make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b
make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b

make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola
make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola
make vanilla_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=128  TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola

# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m DATA_SET=cola
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m DATA_SET=cola
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b DATA_SET=cola
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b
make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola
make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola
make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TEACHER_MODEL_PATH=facebook/opt-2.7b DATA_SET=cola

make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TEACHER_MODEL_PATH=facebook/opt-2.7b
make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TEACHER_MODEL_PATH=facebook/opt-2.7b
make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TEACHER_MODEL_PATH=facebook/opt-2.7b
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=128
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=128
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=128
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=256 BSZ=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=256 BSZ=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=256 BSZ=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m DATA_SET=cola TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m DATA_SET=cola TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b DATA_SET=cola TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b DATA_SET=cola TRAIN_SAMPLE=16 SAVE_MODEL=1
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b TRAIN_SAMPLE=16
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-2.7b TRAIN_SAMPLE=16 SAVE_MODEL=1
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b TRAIN_SAMPLE=16 SAVE_MODEL=1
# make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-6.7b TRAIN_SAMPLE=128 SAVE_MODEL=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m LORA=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m LORA=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b LORA=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-125m BITFIT=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-350m BITFIT=1
#make pattern_verbalizer_ft PACE=1 NUM_GPU=2 MODEL_NAME=facebook/opt-1.3b BITFIT=1

