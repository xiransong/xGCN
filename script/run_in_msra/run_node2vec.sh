PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$4
EMB_TABLE_DEVICE=$DEVICE

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################

p=$5
q=$6
context_size=$7

SEED=1
# RESULTS_DIR="node2vec/[p${p}q${q}][context_size$context_size]"
RESULTS_DIR="node2vec/[best]"

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'node2vec' \
    --config_file $CONFIG_ROOT'/node2vec-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 20 \
    --epochs 200 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --p $p --q $q \
    --context_size $context_size \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
