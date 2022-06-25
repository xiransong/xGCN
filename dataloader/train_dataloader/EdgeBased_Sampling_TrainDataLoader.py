from dataloader.utils import generate_one_strict_neg, generate_multi_strict_neg

import torch
import numpy as np


class EdgeBased_Sampling_TrainDataLoader:
    
    def __init__(self, info, E_src: np.ndarray, E_dst: np.ndarray,
                 batch_size, num_neg=1, ratio=0.1,
                 ensure_neg_is_not_neighbor=False, csr_indptr=None, csr_indices=None,
                 neg_sample_from_active_nodes=False
        ):
        # for each epoch, sample ratio*num_edges edges from all edges
        self.E_src = E_src
        self.E_dst = E_dst
        self.num_edges = len(E_src)
        
        if info['dataset_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
        
        self.batch_size = batch_size
        self.num_neg = num_neg
        
        self.batch_per_epoch = int(self.num_edges * ratio / self.batch_size)
        self.batch_remain = None
        
        self.ensure_neg_is_not_neighbor = ensure_neg_is_not_neighbor
        if self.ensure_neg_is_not_neighbor:
            self.csr_indptr = csr_indptr
            self.csr_indices = csr_indices
        
        self.neg_sample_from_active_nodes = neg_sample_from_active_nodes
        if neg_sample_from_active_nodes:
            all_degrees = csr_indptr[1:] - csr_indptr[:-1]
            self.active_nodes = np.argwhere(all_degrees>0).reshape(-1)
            print("## using neg_sample_from_active_nodes, active nodes:",  self.active_nodes.shape, ", num nodes from indptr:", len(all_degrees)) 

    
    def _generate_strict_neg(self, src):
        if self.num_neg == 1:
            return generate_one_strict_neg(
                self.neg_low, self.neg_high, src, 
                self.csr_indptr, self.csr_indices
            )
        else:
            return generate_multi_strict_neg(
                self.neg_low, self.neg_high, self.num_neg, src, 
                self.csr_indptr, self.csr_indices
            )
    
    def __len__(self):
        return self.batch_per_epoch
    
    def __iter__(self):
        self.batch_remain = self.batch_per_epoch
        return self
    
    def __next__(self):
        if self.batch_remain == 0:
            raise StopIteration
        else:
            self.batch_remain -= 1
        
        eid = torch.randint(0, self.num_edges, (self.batch_size,)).numpy()
        src = torch.LongTensor(self.E_src[eid])
        pos = torch.LongTensor(self.E_dst[eid])
        
        if self.num_neg < 1:
            neg = None
        else:
            if not self.neg_sample_from_active_nodes:
                if self.ensure_neg_is_not_neighbor:
                    neg = torch.LongTensor(self._generate_strict_neg(src.numpy())) 
                else:
                    neg = torch.randint(self.neg_low, self.neg_high, 
                                        (len(src), self.num_neg)).squeeze()
            else:
                neg_ind = np.random.choice(self.active_nodes, size=(len(src), self.num_neg))
                neg = torch.LongTensor(neg_ind)
        
        return src, pos, neg
