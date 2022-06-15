PROJECT_ROOT='/media/data/xGCN' 
ALL_DATA_ROOT='/media/data/xGCN_data'
DATASET='xbox-100m'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets' 
 

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
FILE_INPUT=/media/data/DEV/refactor/xbox/xfriend_auto_pipeline/output_100m/training_instances_xbox_usconsole_directed/graph_csr_RE.pkl

python $PROJECT_ROOT'/'data/handle_csr_graph.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'social' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
