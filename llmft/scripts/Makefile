DATA_SET=mnli
TRAIN_SAMPLE=128
EPOCHS=40
NUM_GPU=1
BSZ=8
MODEL_NAME=facebook/opt-125m
NUM_SHOTS=16
PACE=0
LORA=0
BITFIT=0
TEACHER_MODEL_PATH="NaN"
SAVE_MODEL=0

.PHONY: default pattern_verbalizer_ft vanilla_ft in_context

default: 
	@echo "Pass in a target directly"

# General targets
pattern_verbalizer_ft:
	bash wrapper.sh $(@) $(DATA_SET) $(TRAIN_SAMPLE) $(EPOCHS) 0.5 $(BSZ) $(NUM_GPU) 1e-5 $(MODEL_NAME) 60000 $(PACE) $(LORA) $(BITFIT) $(TEACHER_MODEL_PATH) $(SAVE_MODEL)

vanilla_ft:
	bash wrapper.sh $(@) $(DATA_SET) $(TRAIN_SAMPLE) $(EPOCHS) 0.5 $(BSZ) $(NUM_GPU) 1e-5 $(MODEL_NAME) 60000 $(PACE) $(LORA) $(BITFIT) $(TEACHER_MODEL_PATH) $(SAVE_MODEL)

in_context:
	bash wrapper.sh $(@) $(DATA_SET) $(NUM_SHOTS) $(MODEL_NAME) $(NUM_GPU) 60000 $(PACE) 
