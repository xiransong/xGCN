PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
GAMLP_type='GAMLP_JK'
num_gcn_layers=1
hidden=512
n_layers_1=2
n_layers_2=2
pre_process=0
residual=0
bns=0

RESULTS_DIR="gamlp/[seed$SEED][$GAMLP_type][gcn_layer$num_gcn_layers][h$hidden][n1$n_layers_1][n2$n_layers_2][pre_process$pre_process][residual$residual][bns$bns]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

N2V_EMB='/media/xreco/DEV/xiran/data/social_and_user_item/model_outputs/gnn_xbox-100m/node2vec/saved/out_emb_table.pt'

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/gamlp-config.yaml' \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --train_batch_size 1024 \
    --l2_reg_weight 0.0 \
    --loss_fn 'bpr_loss' \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/test_edges-5000.pkl' --key_score_metric 'n20'  \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges-5000.pkl' \
    --train_batch_size 1024 \
    --emb_dim 32 \
    --epochs 120 --convergence_threshold 10 \
    --edge_sample_ratio 0.01 \
    --from_pretrained 1 --file_pretrained_emb $N2V_EMB \
    --freeze_emb 1 \
    --use_sparse 0 \
    --GAMLP_type $GAMLP_type \
    --num_gcn_layers $num_gcn_layers \
    --hidden $hidden \
    --n_layers_1 $n_layers_1 \
    --n_layers_2 $n_layers_2 \
    --pre_process $pre_process \
    --residual $residual \
    --bns $bns \