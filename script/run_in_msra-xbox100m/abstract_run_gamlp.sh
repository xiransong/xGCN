source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

num_gcn_layers=$1

bash $PROJECT_ROOT/script/run_in_msra-xbox100m/run_gamlp.sh $PROJECT_ROOT $ALL_DATA_ROOT $num_gcn_layers
