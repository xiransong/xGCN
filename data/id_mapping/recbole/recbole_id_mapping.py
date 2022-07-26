import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils import xconfig
from utils.utils import wc_count, ensure_dir, ReIndexDict

import numpy as np
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = xconfig.xConfig()
    config.parse(sys.argv[2:])
    config.print()
    
    data_root = config['data_root']
    ensure_dir(data_root)
    file_input = config['file_input']
    
    num_edges = wc_count(file_input) - 1
    E = np.empty((2, num_edges), dtype=np.int32)
    
    user_dict = ReIndexDict()  # re index, map old id to new id
    item_dict = ReIndexDict()
    
    print("load and reindex from .txt")
    with open(file_input, 'r') as f:
        f.readline()  # skip the first line
        for i in tqdm(range(num_edges)):
            line = f.readline().split()
            E[0][i] = user_dict[line[0]]
            E[1][i] = item_dict[line[1]]
    
    num_users = len(user_dict)
    num_items = len(item_dict)
    E[1] += num_users
    io.save_pickle(osp.join(data_root, 'edges.pkl'), E)
    del E
    
    info = {
        'dataset_type': 'user-item',
        'num_nodes': num_users + num_items,
        'num_users': num_users,
        'num_items': num_items,
        'num_edges': num_edges
    }
    io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    
    dic_root = osp.join(data_root, 'node_id_map')
    ensure_dir(dic_root)
    io.save_pickle(osp.join(dic_root, 'user_old2new_dict.pkl'), user_dict.get_old2new_dict())
    io.save_pickle(osp.join(dic_root, 'user_new2old_list.pkl'), user_dict.get_new2old_list())
    io.save_pickle(osp.join(dic_root, 'item_old2new_dict.pkl'), item_dict.get_old2new_dict())
    io.save_pickle(osp.join(dic_root, 'item_new2old_list.pkl'), item_dict.get_new2old_list())


if __name__ == '__main__':
    
    setproctitle.setproctitle('id_mapping-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
