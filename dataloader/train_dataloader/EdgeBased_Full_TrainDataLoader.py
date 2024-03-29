from dataloader.utils import generate_one_strict_neg, generate_multi_strict_neg

import torch
import numpy as np


class EdgeBased_Full_TrainDataLoader:
    
    def __init__(self, info, E_src: np.ndarray, E_dst: np.ndarray,
                 batch_size, num_neg=1,
                 ensure_neg_is_not_neighbor=False, csr_indptr=None, csr_indices=None, 
                 use_degree_for_neg_sample=False, undi_indptr=None):
        self.dl = torch.utils.data.DataLoader(
            dataset=torch.stack([
                torch.LongTensor(E_src), 
                torch.LongTensor(E_dst)]).T,
            batch_size=batch_size,
            shuffle=True
        )
        self.num_neg = num_neg
        
        if info['dataset_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
            
        self.ensure_neg_is_not_neighbor = ensure_neg_is_not_neighbor
        if self.ensure_neg_is_not_neighbor:
            self.csr_indptr = csr_indptr
            self.csr_indices = csr_indices
        
        self.use_degree_for_neg_sample = use_degree_for_neg_sample
        self.samples_weights = None
        if self.use_degree_for_neg_sample:
            print("## use_degree_for_neg_sample")
            all_degrees = undi_indptr[1:] - undi_indptr[:-1]
            self.sample_weights = torch.FloatTensor(all_degrees[self.neg_low:self.neg_high])
            self.sample_weights = self.sample_weights ** 0.75
        
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
            
    def sample_node_by_degree(self, N):
        nids = torch.multinomial(self.sample_weights, num_samples=N, replacement=True)
        nids += self.neg_low
        return nids
       
    def get_neg_samples(self, src):
        if self.use_degree_for_neg_sample:
            neg = self.sample_node_by_degree(len(src) * self.num_neg).view(len(src), self.num_neg)
        else:
            if self.ensure_neg_is_not_neighbor:
                neg = torch.LongTensor(self._generate_strict_neg(src.numpy()))
            else:
                neg = torch.randint(self.neg_low, self.neg_high, 
                                    (len(src), self.num_neg)).squeeze()
        return neg
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        self.dl_iter = iter(self.dl)
        return self
    
    def __next__(self):
        src_pos = next(self.dl_iter)
        src, pos = src_pos[:,0], src_pos[:,1]
        
        if self.num_neg < 1:
            neg = None
        else:
            neg = self.get_neg_samples(src)
        
        return src, pos, neg
