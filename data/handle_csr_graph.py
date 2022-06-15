import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, wc_count
from data.handle_train_graph import handle_train_graph

import numpy as np
from tqdm import tqdm
import setproctitle
import time
import pickle as pkl
import os
from datetime import datetime

def load_pkl_obj(filename):
    print('INFO: loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, "rb") as f: 
        obj = pkl.load(f)  
        
    print('finish loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return obj

def load_csr_graph_as_edge_list(filename):
    graph = load_pkl_obj(filename)
    indices, indptr = graph['nei_array'], graph['ptr_array'] 
    num_lines = wc_count(filename) 
    N, M = len(indptr)-1, len(indices)
    E_src = np.zeros(M, dtype=np.int32)
    E_dst =  np.zeros(M, dtype=np.int32)
    cnt = 0
    for i in tqdm(range(N), desc='convert to edge array'):
        if indptr[i+1] > indptr[i]:
            for j in range(indptr[i], indptr[i+1]):
                E_src[cnt] = i
                E_dst[cnt] = indices[j]
                cnt += 1
    assert M == cnt
    return E_src, E_dst


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']  # place to save the processed data
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    assert dataset_type in ['user-item', 'social']
    
    file_input = config['file_input']
    
    print("## load cse graph", file_input)
    E_src, E_dst = load_csr_graph_as_edge_list(file_input)
   
    handle_train_graph(E_src, E_dst, data_root, dataset_name, dataset_type)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_adj_graph_txt-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
