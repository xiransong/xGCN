PROJECT_ROOT='/home/sxr/code/xgcn'

# edges
INPUT_DATA_ROOT='/home/sxr/data/social_and_user_item/raw_datasets/MovieLens-10m/processed/'

# dir to save the train/eval edges
DATASET_NAME='MovieLens-10m'
OUTPUT_DATA_ROOT='/home/sxr/data/social_and_user_item/datasets/instance_'$DATASET_NAME

SEED=2022

python split_pos_edges.py $PROJECT_ROOT \
    'seed:int:'$SEED \
    'input_data_root:str:'$INPUT_DATA_ROOT \
    'output_data_root:str:'$OUTPUT_DATA_ROOT \
    'num_val:int:'5000 \
    'num_test:int:'50000 \
    'min_src_out_degree:int:'3 \
    'min_dst_in_degree:int':3 \
