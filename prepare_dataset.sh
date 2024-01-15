export DATASET_DIR=~/m4t_dataset
mkdir -p $DATASET_DIR

m4t_prepare_dataset \
  --source_lang eng \
  --target_lang kor \
  --split train \
  --save_dir $DATASET_DIR
m4t_prepare_dataset \
  --source_lang eng \
  --target_lang kor \
  --split validation \
  --save_dir $DATASET_DIR