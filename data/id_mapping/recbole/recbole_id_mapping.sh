PROJECT_ROOT='/home/sxr/code/xgcn'

RAW_DATA_ROOT='/home/sxr/data/social_and_user_item/raw_datasets/MovieLens-10m'

FILE_INPUT=$RAW_DATA_ROOT'/ml-10m.inter'
DATA_ROOT=$RAW_DATA_ROOT'/processed'

python recbole_id_mapping.py $PROJECT_ROOT \
    'data_root:str:'$DATA_ROOT \
    'file_input:str:'$FILE_INPUT \
