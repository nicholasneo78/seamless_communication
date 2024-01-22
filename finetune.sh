

# before running this run 
# ./shift_yaml.sh
# ./prepare_dataset.sh

# torchrun \
#    --rdzv-backend=c10d \
#    --rdzv-endpoint=localhost:0 \
#    --nnodes=1 \
#    --nproc-per-node=1 \
#    --no-python \

# export DATASET_DIR=~/m4t_dataset
# mkdir -p $DATASET_DIR

# m4t_prepare_dataset \
#   --source_lang eng \
#   --target_lang kor \
#   --split train \
#   --save_dir $DATASET_DIR
# m4t_prepare_dataset \
#   --source_lang eng \
#   --target_lang kor \
#   --split validation \
#   --save_dir $DATASET_DIR

# learning rate of 2e-5 is the default

m4t_finetune \
--mode SPEECH_TO_TEXT \
--model_name seamlessM4T_medium \
--train_dataset /seamless_communication/m4t_dataset/train_manifest.json \
--eval_dataset /seamless_communication/m4t_dataset/dev_manifest.json \
--batch_size 2 \
--learning_rate 2e-6 \
--warmup_steps 2330 \
--max_epochs 2 \
--patience 10 \
--eval_steps 1165 \
--log_steps 100 \
--save_model_to /models/m4t_finetuned/checkpoint.pt