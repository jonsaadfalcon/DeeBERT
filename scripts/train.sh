#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=glue_data

MODEL_TYPE=roberta  # bert or roberta
MODEL_SIZE=large  # base or large
DATASET=MRPC  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
CHOSEN_DATASET=sciie-relation-extraction #sci-cite #sciie-relation-extraction #chemprot #sci-cite
NUMBER_OF_LABELS=7

#MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
MODEL_NAME=roberta-large #allenai/scibert_scivocab_uncased #roberta-large
EPOCHS=10
LEARNING_RATE=1e-5
FROZEN_LAYERS=0


python -um examples.run_text_classification_baseline_highway \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --text_classification_dataset $CHOSEN_DATASET \
  --num_labels $NUMBER_OF_LABELS \
  --frozen_layers $FROZEN_LAYERS \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $EPOCHS \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ./saved_models/baseline/${MODEL_NAME}-${MODEL_SIZE}/$CHOSEN_DATASET/$FROZEN_LAYERS/$LEARNING_RATE/two_stage \
  --save_steps 0 \
  --overwrite_cache
