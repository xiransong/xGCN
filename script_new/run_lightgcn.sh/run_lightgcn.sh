PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

DATASET='MovieLens-1m'
DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

ITEM_FEAT_FIELD='EmbeddingMultiplexer|item_tower|feat_emb_tables'

FEAT_ROOT=$DATA_ROOT'/processed_feat'

FEAT_MOVIE_TITLE=$FEAT_ROOT'/item-300d/movie_title-token_seq.pkl'
FEAT_MOVIE_CAT=$FEAT_ROOT'/item-300d/genre-token_seq.pkl'

RESULTS_DIR='[1]'
RESULRS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/lightgcn/'$RESULTS_DIR

python $PROJECT_ROOT/main/main.py $PROJECT_ROOT 'block_lightgcn.yaml' \
    'data_root:str:'$DATA_ROOT \
    'results_root:str:'$RESULRS_ROOT \
    'seed:int:'1 \
    'eval|file_validation:str:'$DATA_ROOT'/val_edges.pkl' \
    'eval|file_test:str:'$DATA_ROOT'/test_edges.pkl' \
    $ITEM_FEAT_FIELD'|movie_title|file_pretrained_emb:str:'$FEAT_MOVIE_TITLE \
    $ITEM_FEAT_FIELD'|movie_category|file_pretrained_emb:str:'$FEAT_MOVIE_CAT \
