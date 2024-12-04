# Wraps around the deepspeed scripts 
# vanilla/pbft args: ft_name, task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port, pace, lora, bitfit, teacher_model_path, save_model
# incontext args: ft_name, task_name, num_shots, model_name_or_path, gpu, port, pace
ft_name=$1
task_name=$2

if [[ "$ft_name" == "in_context" ]]
then
num_gpu=$5
pace=$7
else
num_gpu=$7
pace=${11}
fi

cuda_dev=""
for ((i=0 ; i < $num_gpu ; i++))
do
    if [[ i -lt $(($num_gpu-1)) ]]
    then
	    cuda_dev+=$i,
    else 
	    cuda_dev+=$i
    fi 
done
export CUDA_VISIBLE_DEVICES=$cuda_dev

echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

if [[ $pace -eq 1 ]]
then
export PROJECT_DIR=~/Neural-Network-Ninjas/llmft
source $PROJECT_DIR/scripts/misc/setup_pace.sh
else
export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh
fi

# deepspeed args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
# deepspeed is just offset by 1
if [[ "$ft_name" == "in_context" ]]
then
    bash $PROJECT_DIR/scripts/$ft_name/$task_name/run_minimal.sh $task_name $3 $4 $5 $6
    bash $PROJECT_DIR/scripts/$ft_name/$task_name/run_gpt3.sh $task_name $3 $4 $5 $6
    bash $PROJECT_DIR/scripts/$ft_name/$task_name/run_eval_harness.sh $task_name $3 $4 $5 $6
else
    bash $PROJECT_DIR/scripts/$ft_name/$task_name/run.sh $task_name $3 $4 $5 $6 $7 $8 $9 ${10} ${14} ${15}
fi
