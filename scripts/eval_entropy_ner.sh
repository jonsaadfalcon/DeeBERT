
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=glue_data

MODEL_TYPE=roberta  # bert or roberta
MODEL_SIZE=large  # base or large
DATASET=MRPC  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

CHOSEN_DATASET=NCBI-disease  #bc5cdr JNLPBA  NCBI-disease 
NUMBER_OF_LABELS=3 #3, 11, ,3

#MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
MODEL_NAME=roberta-large #allenai/scibert_scivocab_uncased #roberta-large
EPOCHS=10
LEARNING_RATE=0.00002
FROZEN_LAYERS=0
RUNS=1

ENTROPIES="0 0.2 0.4 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4"

for ENTROPY in $ENTROPIES; do
  echo $ENTROPY
  python -um examples.run_ner_highway \
    --model_type $MODEL_TYPE \
    --model_name_or_path ./saved_models/${MODEL_NAME}-${MODEL_SIZE}/$CHOSEN_DATASET/$FROZEN_LAYERS/$LEARNING_RATE/two_stage \
    --task_name $DATASET \
    --text_classification_dataset $CHOSEN_DATASET \
    --num_labels $NUMBER_OF_LABELS \
    --frozen_layers $FROZEN_LAYERS \
    --runs $RUNS \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --output_dir ./saved_models/${MODEL_NAME}-${MODEL_SIZE}/$CHOSEN_DATASET/$FROZEN_LAYERS/$LEARNING_RATE/two_stage \
    --plot_data_dir ./plotting/ \
    --max_seq_length 128 \
    --early_exit_entropy $ENTROPY \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1
done