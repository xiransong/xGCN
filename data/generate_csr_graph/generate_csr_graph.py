import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from data.csr_graph_helper import get_src_indices, from_edges_to_csr, remove_repeated_edges_in_csr
from utils import io
from utils import xconfig

import numpy as np
import setproctitle
import time


def main():
    
    config = xconfig.xConfig()
    config.parse(sys.argv[2:])
    config.print()
    
    E = io.load_pickle(config['file_input'])  # numpy array, shape=(2, num_edges)
    num_nodes = E.max() + 1
    
    print("# construct csr graph ...")
    print("- from_edges_to_csr ...")
    indptr, indices = from_edges_to_csr(E[0], E[1], num_nodes)
    
    print("- remove_repeated_edges_in_csr ...")
    indptr, indices = remove_repeated_edges_in_csr(indptr, indices)
    
    print("- get_src_indices ...")
    src_indices = get_src_indices(indptr)
    
    print("- save csr graph ...")
    io.save_pickle(config['file_csr_indptr'], indptr)
    io.save_pickle(config['file_csr_indices'], indices)
    io.save_pickle(config['file_csr_src_indices'], src_indices)
    
    if config['generate_undirected']:
        print("# construct undirected csr graph ...")
        undi_E_src = np.concatenate([src_indices, indices])
        undi_E_dst = np.concatenate([indices, src_indices])
        
        del indptr, indices, src_indices
        
        print("- from_edges_to_csr ...")
        undi_indptr, undi_indices = from_edges_to_csr(undi_E_src, undi_E_dst, num_nodes)
        
        del undi_E_src, undi_E_dst
        
        print("- remove_repeated_edges_in_csr ...")
        undi_indptr, undi_indices = remove_repeated_edges_in_csr(undi_indptr, undi_indices)
        
        print("- get_src_indices ...")
        undi_src_indices = get_src_indices(undi_indptr)
        
        print("- save csr graph ...")
        io.save_pickle(config['file_undi_csr_indptr'], undi_indptr)
        io.save_pickle(config['file_undi_csr_indices'], undi_indices)
        io.save_pickle(config['file_undi_csr_src_indices'], undi_src_indices)

    print("# done!")


if __name__ == '__main__':
    
    setproctitle.setproctitle('generate_csr_graph-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
