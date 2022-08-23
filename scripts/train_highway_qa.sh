export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=glue_data

MODEL_TYPE=roberta  # bert or roberta
MODEL_SIZE=large  # base or large
DATASET=MRPC  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

CHOSEN_DATASET=trivia_qa #SQuAD or triviaqa 
NUMBER_OF_LABELS=2 #3, 11, 3,

#MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
MODEL_NAME=roberta-large #allenai/scibert_scivocab_uncased #roberta-large
EPOCHS=4
FROZEN_LAYERS=0
RUNS=1

#LEARNING_RATES="0.0001 0.0002 0.00001 0.00002 0.00005 0.000005"
LEARNING_RATES="0.00002"

for LEARNING_RATE in $LEARNING_RATES; do
  echo $CHOSEN_LEARNING_RATE
  python -um examples.run_qa_highway \
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
    --runs $RUNS \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --overwrite_output_dir \
    --seed 42 \
    --output_dir /net/nfs.cirrascale/s2-research/jons/trivia_qa/saved_models/${MODEL_NAME}-${MODEL_SIZE}/$CHOSEN_DATASET/$FROZEN_LAYERS/$LEARNING_RATE/two_stage \
    --plot_data_dir ./plotting/ \
    --save_steps 1000 \
    --overwrite_cache \
    --eval_after_first_stage
done

