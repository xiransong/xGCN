PROJECT_ROOT='/media/data/xGCN' 
ALL_DATA_ROOT='/media/data/xGCN_data'
DATASET='xbox-100m-recent'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets' 
 

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
FILE_INPUT=/media/data/xGCN_data/datasets/rawdata-xbox-100m/recent_graph_csr_RE.pkl

python $PROJECT_ROOT'/'data/handle_csr_graph.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'social' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
