PROJECT_ROOT='/media/data/xGCN' 
ALL_DATA_ROOT='/media/data/xGCN_data'
DATASET='xbox-100m'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

# input format: src \t dst \t label
# each src has 1 pos and k neg
# [[src01, dst01, 1], 
#  [src01, dst02, 0], 
#   ..., 
#  [src01, dst0k, 0],
#  [src02, dst01, 1], 
#  [src02, dst02, 0], 
#   ..., 
#  [src02, dst0k, 0], ...]

# output: numpy array, [[src01, pos, neg1, ..., negk], [src02, ... ], ... ]


DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python data/handle_labelled_eval_set.py $PROJECT_ROOT \
    --file_input '/media/data/DEV/refactor/xbox/xfriend_auto_pipeline/output_100m/training_instances_xbox_usconsole_directed/valid' \
    --file_output $DATA_ROOT'/valid-1-99.pkl' \
    --file_output_2 $DATA_ROOT'/valid-1-99-pos_edges.pkl' \


python data/handle_labelled_eval_set.py $PROJECT_ROOT \
    --file_input '/media/data/DEV/refactor/xbox/xfriend_auto_pipeline/output_100m/training_instances_xbox_usconsole_directed/test' \
    --file_output $DATA_ROOT'/test-1-99.pkl' \
    --file_output_2 $DATA_ROOT'/test-1-99-pos_edges.pkl' \

###### for model training:
#   --validation_method 'one_pos_k_neg' \
#   --mask_nei_when_validation 0 \
#   --file_validation $DATA_ROOT'/test-1-99.pkl' \
#   --test_method 'one_pos_k_neg' \
#   --mask_nei_when_test 0 \
#   --file_test $DATA_ROOT'/test-1-99.pkl' \
######
