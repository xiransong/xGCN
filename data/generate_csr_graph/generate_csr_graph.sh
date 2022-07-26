PROJECT_ROOT='/home/sxr/code/xgcn'

DATASET_NAME='MovieLens-10m'
DATA_ROOT='/home/sxr/data/social_and_user_item/datasets/instance_'$DATASET_NAME

FILE_INPUT=$DATA_ROOT'/train_edges.pkl'

python generate_csr_graph.py $PROJECT_ROOT \
    'file_input:str:'$FILE_INPUT \
    'generate_undirected:bool:True' \
    'file_csr_indptr:str:'$DATA_ROOT'/train_csr_indptr.pkl' \
    'file_csr_indices:str:'$DATA_ROOT'/train_csr_indices.pkl' \
    'file_csr_src_indices:str:'$DATA_ROOT'/train_csr_src_indices.pkl' \
    'file_undi_csr_indptr:str:'$DATA_ROOT'/train_undi_indptr.pkl' \
    'file_undi_csr_indices:str:'$DATA_ROOT'/train_undi_csr_indices.pkl' \
    'file_undi_csr_src_indices:str:'$DATA_ROOT'/train_undi_csr_src_indices.pkl' \
