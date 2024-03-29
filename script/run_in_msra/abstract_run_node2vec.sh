source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

# PROJECT_ROOT='/home/sxr/code/www23/xGCN'
# ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

DEVICE='cuda'

########################################
DATASET=$1

p=$2
q=$3
context_size=$4

# $PROJECT_ROOT/script/run_in_msra/run_node2vec.sh

bash xGCN/script/run_in_msra/run_node2vec.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE \
    $DATASET $p $q $context_size \
