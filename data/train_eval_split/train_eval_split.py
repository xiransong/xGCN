import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils import xconfig
from utils.utils import ensure_dir

import numpy as np
import torch
import dgl
from copy import deepcopy
import os.path as osp
import setproctitle
import time


def main():
    
    config = xconfig.xConfig()
    config.parse(sys.argv[2:])
    config.print()
    
    input_data_root = config['input_data_root']
    output_data_root = config['output_data_root']
    ensure_dir(output_data_root)
    np.random.seed(config['seed'])
    
    raw_graph_info = io.load_yaml(osp.join(input_data_root, 'info.yaml'))
    dataset_type = raw_graph_info['dataset_type']
    
    E = io.load_pickle(osp.join(input_data_root, 'edges.pkl'))
    # numpy array, shape=(2, num_edges)
    g = dgl.graph((
        torch.LongTensor(E[0]),
        torch.LongTensor(E[1])
    ))
    del E
    print(g)
    
    in_degrees = deepcopy(g.in_degrees().numpy())
    out_degrees = deepcopy(g.out_degrees().numpy())
    
    num_val = config['num_val']
    num_test = config['num_test']
    num_sample = num_val + num_test
    
    min_src_out_degree = config['min_src_out_degree']
    min_dst_in_degree = config['min_dst_in_degree']
    
    def src_degree_ok(node):
        return (min_src_out_degree < out_degrees[node])
    
    def dst_degree_ok(node):
        return (min_dst_in_degree < in_degrees[node])
    
    all_nodes = np.arange(g.num_nodes())
    pos_edges = []
    while True:
        np.random.shuffle(all_nodes)
        exists_ok_node = False
        for s in all_nodes:
            print("sampling edges {}/{}".format(len(pos_edges), num_sample), end='\r')
            if src_degree_ok(s):
                nei = deepcopy(g.successors(s).numpy())  # must use deepcopy, otherwise the graph will be damaged
                np.random.shuffle(nei)
                for d in nei:
                    if dst_degree_ok(d):
                        exists_ok_node = True
                        out_degrees[s] -= 1
                        in_degrees[d] -= 1
                        pos_edges.append((s, d))
                        break
                if len(pos_edges) >= num_sample:
                    break
        if len(pos_edges) >= num_sample or not exists_ok_node:
            break
    print("\nnum sampled edges:", len(pos_edges))
    pos_edges = np.array(pos_edges)
    
    print("remove pos edges from g")
    g.remove_edges(  # remove pos edges (in-place)
        g.edge_ids(pos_edges[:,0], pos_edges[:,1])
    )
    train_graph = g
    
    if dataset_type == 'user-item':
        pos_edges[:,1] -= raw_graph_info['num_users']
    
    val_edges = pos_edges[:num_val]
    test_edges = pos_edges[num_val:]
    io.save_pickle(osp.join(output_data_root, 'val_edges.pkl'), val_edges)
    io.save_pickle(osp.join(output_data_root, 'test_edges.pkl'), test_edges)
    
    E = g.edges()
    E = np.stack([
        np.array(E[0].numpy(), dtype=np.int32),
        np.array(E[1].numpy(), dtype=np.int32)
    ])
    io.save_pickle(osp.join(output_data_root, 'train_edges.pkl'), E)
    
    cf = config.get_dict()
    cf['num_val_edges'] = len(val_edges)
    cf['num_test_edges'] = len(test_edges)
    cf['num_train_edges'] = g.num_edges()
    io.save_yaml(osp.join(output_data_root, 'config-train_eval_split.yaml'), cf)
    
    info = {
        'dataset_type': dataset_type,
        'num_nodes': train_graph.num_nodes(),
        'num_users': raw_graph_info['num_users'],
        'num_items': raw_graph_info['num_items'],
        'num_edges': train_graph.num_edges()
    }
    io.save_yaml(osp.join(output_data_root, 'info.yaml'), info)


if __name__ == '__main__':
    
    setproctitle.setproctitle('train_eval_split-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
