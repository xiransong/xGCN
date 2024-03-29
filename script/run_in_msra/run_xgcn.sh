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
SEED=1
T=$5
# K=5
K=$6
endure=3
# endure=$7
# convergence_threshold=$8
# max_prop_times=$9

use_special_dnn=$7
dnn_arch=$8

# RESULTS_DIR="xgcn/[seed${SEED}][K${K}][endure${endure}][max_prop_times${max_prop_times}]"
RESULTS_DIR="xgcn/ablation/[scale${use_special_dnn}][dnn_arch$dnn_arch]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

mkdir -p $RESULTS_ROOT
LOG_FILE=$RESULTS_ROOT'/log.txt'

echo "[begin at $(date)]" > $LOG_FILE
echo "results dir: "$RESULTS_ROOT

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'xgcn' \
    --config_file $CONFIG_ROOT'/xgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --emb_table_device $EMB_TABLE_DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 30 \
    --epochs 300 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --prop_type 'lightgcn' --num_gcn_layers 1 \
    --use_special_dnn $use_special_dnn --dnn_arch $dnn_arch \
    --renew_and_prop_freq $T --K $K --endure $endure \
    --emb_init_std 1.0 \
    # --max_prop_times $max_prop_times \
    # --use_special_dnn 1 --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
    # >> $LOG_FILE

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
echo "[end at $(date)]" >> $LOG_FILE

# [nn.Linear(64,1024),nn.Tanh(),nn.Linear(1024,1024),nn.Tanh(),nn.Linear(1024,64)]