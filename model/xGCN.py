import numpy as np
import torch
import torch.nn as nn
import dgl
import os.path as osp
from copy import deepcopy

from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product, bpr_loss, bce_loss, cosine_contrastive_loss
from model.LightGCN import get_lightgcn_out_emb
from data.csr_graph_helper import numba_csr_mult_dense
from helper.eval_helper import eval_model


class MyDNN(torch.nn.Module):
    
    def __init__(self, dnn_arch, scale_net_arch):
        super(MyDNN, self).__init__()
        self.dnn = torch.nn.Sequential(*eval(dnn_arch))
        # self.dnn = torch.nn.Sequential(
        #     torch.nn.Linear(64, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 64)
        # )
        
        self.scale_net = torch.nn.Sequential(*eval(scale_net_arch))
        # self.scale_net = torch.nn.Sequential(
        #     torch.nn.Linear(64, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Sigmoid()
        # )
    
    def forward(self, X):
        
        theta = self.scale_net(X)
        
        X = theta * self.dnn(X)
        
        return X


def train_identical_mapping_dnn(dnn, embeddings):
    print("## train_identical_mapping_dnn")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(dnn.parameters(), lr=0.001)
    
    def get_loss():
        idx = torch.randint(0, len(embeddings), (1024,))
        X = embeddings[idx]
        X_out = dnn(X)
        loss = loss_fn(X, X_out)
        return loss
    
    best_loss = 999999
    with torch.no_grad():
        loss = get_loss()
    epoch = 0
    
    while True:
        if not (epoch % 100):
            print(epoch, loss.item())
            if loss.item() > best_loss:
                break
            best_loss = loss.item()
        epoch += 1
        
        opt.zero_grad()
        loss = get_loss()
        loss.backward()
        opt.step()
    
    return dnn


class xGCN(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']  # for dnn
        self.emb_table_device = self.config['emb_table_device']  # for embeddings
        
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        self.num_nodes = self.info['num_nodes']
        
        data_root = self.config['data_root']
        
        if 'gamma' in self.config and self.config['gamma'] > 0:
            print("## using item-item graph additional training signal")
            self.ii_topk_neighbors = io.load_pickle(config['file_ii_topk_neighbors'])
            # self.ii_topk_similarity_scores = io.load_pickle(config['file_ii_topk_similarity_scores'])
            
            topk = config['topk']
            self.ii_topk_neighbors = torch.LongTensor(self.ii_topk_neighbors[:, :topk]).to(self.device)
            # self.ii_topk_similarity_scores = torch.FloatTensor(self.ii_topk_similarity_scores[:, :topk]).to(self.device)

        self.prop_type = self.config['prop_type']
        if self.prop_type == 'pprgo':
            print("## load ppr neighbors and ppr weights ...")
            raw_nei = io.load_pickle(osp.join(config['ppr_data_root'], "nei.pkl"))
            raw_wei = io.load_pickle(osp.join(config['ppr_data_root'], "wei.pkl"))
            
            topk = config['topk']
            self.nei = torch.LongTensor(raw_nei[:, : topk])
            self.wei = torch.FloatTensor(raw_wei[:, : topk])
            
            if config['use_uniform_weight']:
                print("## uniform weight")
                _w = torch.ones(self.nei.shape)
                _w[self.wei == 0] = 0
                self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                print("## not uniform weight")
                self.wei = self.wei / (self.wei.sum(dim=-1, keepdim=True) + 1e-12)
            
        else:
            assert self.prop_type == 'lightgcn'
            
            if 'use_item2item_graph_for_item_prop' in self.config and \
                self.config['use_item2item_graph_for_item_prop']:
                topk = self.config['topk']
                ii_topk_neighbors = io.load_pickle(config['file_ii_topk_neighbors'])
                self.ii_topk_neighbors = torch.LongTensor(ii_topk_neighbors[:, :topk])
                
            E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
            E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
            indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
            all_degrees = indptr[1:] - indptr[:-1]
            d_src = all_degrees[E_src]
            d_dst = all_degrees[E_dst]
            
            edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()
            del d_src, d_dst
            
            if 'zero_degree_zero_emb' in self.config and self.config['zero_degree_zero_emb']:
                self.zero_degree_nodes = all_degrees == 0
                self.num_zero_degree = self.zero_degree_nodes.sum().item()
                print("## using zero_degree_zero_emb, num_zero_degree:", 
                      self.zero_degree_nodes.sum().item(), ", num_nodes:", self.num_nodes)
            del all_degrees
            
            if self.config['use_numba_csr_mult']:
                self.indptr = indptr
                self.indices = E_dst
                self.edge_weights = edge_weights.numpy()
            elif hasattr(torch, 'sparse_csr_tensor'):
                self.A = torch.sparse_csr_tensor(
                    torch.LongTensor(np.array(indptr, dtype=np.int64)),
                    torch.LongTensor(E_dst),
                    edge_weights,
                    (self.num_nodes, self.num_nodes)
                )
            else:
                del indptr
                E_src = torch.IntTensor(E_src)
                E_dst = torch.IntTensor(E_dst)
                E = torch.cat([E_src, E_dst]).reshape(2, -1)
                del E_src, E_dst
                self.A = torch.sparse_coo_tensor(
                    E, edge_weights, 
                    (self.num_nodes, self.num_nodes)
                )
        
        self.emb_table = init_emb_table(config, self.num_nodes)
        self.emb_table = self.emb_table.weight
        
        # build dnn and optimizer
        if self.config['use_two_dnn']:
            assert self.dataset_type == 'user-item'
            self.user_dnn = self._build_dnn()
            self.item_dnn = self._build_dnn()
            dnn_params = [*self.user_dnn.parameters(), *self.item_dnn.parameters()]
        else:
            self.dnn = self._build_dnn()
            dnn_params = self.dnn.parameters()
        
        self.param_list = []
        self.param_list.append({'params': dnn_params, 'lr': self.config['dnn_lr'],
                                'weight_decay': self.config['dnn_l2_reg_weight']})

        # renew/propagation config
        # self.renew_and_prop_freq = self.config['renew_and_prop_freq']
        # self.epoch = 0
        
        # self.max_renew_times = self.config['max_renew_times']
        # self.already_done_renew_times = 0
        
        if 'max_prop_times' in self.config:
            self.max_prop_times = self.config['max_prop_times']
        # self.already_done_prop_times = 0
        
        self.epoch_last_prop = 0
        self.total_prop_times = 0
        
        if not self.config['from_pretrained']:
            self._do_propagation()
    
    def parameters(self):
        return self.param_list

    def _build_dnn(self):
        if 'use_special_dnn' in self.config and self.config['use_special_dnn']:
            print("## using scale-dnn")
            dnn = MyDNN(self.config['dnn_arch'], self.config['scale_net_arch']).to(self.device)
        else:
            dnn = torch.nn.Sequential(*eval(self.config['dnn_arch'])).to(self.device)
        
        if 'use_identical_dnn' in self.config and self.config['use_identical_dnn']:
            dnn = train_identical_mapping_dnn(dnn, self.emb_table)
        return dnn
    
    def _print_emb_info(self, table, name):
        emb_abs = table.abs()
        print("[{} info]:".format(name))
        print("    .abs().max():", emb_abs.max())
        print("    .abs().mean():", emb_abs.mean())
        print("    .std():", table.std())
    
    def _calc_pprgo_out_emb(self, nids):
        top_nids = self.nei[nids]
        top_weights = self.wei[nids]
        
        top_embs = self.emb_table[top_nids].to(self.device)
        top_weights = top_weights.to(self.device)
        
        out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        return out_embs.squeeze()
    
    def _calc_item2item_emb(self, item_nids):
        top_nids = self.ii_topk_neighbors[item_nids]
        return self.emb_table[top_nids + self.num_users].mean(dim=-2)
    
    def _do_propagation(self):
        if 'cancel_prop' in self.config and self.config['cancel_prop']:
            # do nothing
            return
        
        with torch.no_grad():
            print("## do_propagation:", self.prop_type)
            if self.prop_type == 'pprgo':
                _emb_table = torch.empty(self.emb_table.shape, dtype=torch.float32).to(self.emb_table_device)
                dl = torch.utils.data.DataLoader(dataset=torch.arange(len(self.emb_table)), 
                                                 batch_size=8192)
                for nids in dl:
                    _emb_table[nids] = self._calc_pprgo_out_emb(nids).to(self.emb_table_device)
                
                self.emb_table = _emb_table
                
            else:  # self.prop_type == 'lightgcn'
                
                if 'use_item2item_graph_for_item_prop' in self.config and \
                    self.config['use_item2item_graph_for_item_prop']:
                    print("## use_item2item_graph_for_item_prop")
                    
                    self.emb_table = self.emb_table.cpu()
                    _emb_table = torch.empty(self.emb_table.shape, dtype=torch.float32)
                    
                    dl = torch.utils.data.DataLoader(
                        dataset=torch.arange(self.info['num_items']), 
                        batch_size=8192
                    )
                    for item_nids in dl:
                        _emb_table[item_nids + self.num_users] = self._calc_item2item_emb(item_nids)
                    
                    X = get_lightgcn_out_emb(
                        self.A, self.emb_table, self.config['num_gcn_layers'],
                        stack_layers=self.config['stack_layers']
                    )
                    _emb_table[:self.num_users] = X[:self.num_users]
                    
                    self.emb_table = _emb_table.to(self.emb_table_device)
                    
                else:
                    if self.config['use_numba_csr_mult']:
                        print("- use_numba_csr_mult, do not stack")
                        X_in = self.emb_table.cpu().numpy()
                        X_out = np.empty(X_in.shape, dtype=np.float32)
                        numba_csr_mult_dense(
                            self.indptr, self.indices, self.edge_weights,
                            X_in, X_out
                        )
                        self.emb_table = torch.FloatTensor(X_out).to(self.emb_table_device)
                        del X_in
                    else:
                        self.emb_table = get_lightgcn_out_emb(
                            self.A, self.emb_table.cpu(), self.config['num_gcn_layers'],
                            stack_layers=self.config['stack_layers']
                        ).to(self.emb_table_device)
            self._print_emb_info(self.emb_table, 'emb_table')
        
        self.out_emb_table = self.emb_table
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
        
        if 'val_dl' in self.data:
            with torch.no_grad():
                results = eval_model(self, self.data['val_dl'], desc='val')
                print("## after prop:", results)
                record_file = osp.join(self.config['results_root'], 'train_record.txt')
                with open(record_file, "a") as f:
                    f.write('after prop:')
                    f.write(
                        ','.join(map(
                            lambda x: "{:.4g}".format(x) if isinstance(x, float) else str(x), 
                            results.values())
                        ) + '\n'
                    )
    
    def _infer_dnn_output_emb(self, dnn, input_table, output_table):
        with torch.no_grad():
            # _dnn = deepcopy(dnn).to(input_table.device)
            dl = torch.utils.data.DataLoader(
                torch.arange(len(input_table)),
                batch_size=4096
            )
            for idx in dl:
                # output_table[idx] = _dnn(input_table[idx]).to(output_table.device)
                output_table[idx] = self.get_output_emb(None, dnn, input_table[idx])
    
    def get_output_emb(self, nids, dnn, input_emb=None):
        if input_emb is None:
            input_emb = self.emb_table[nids]
        if 'use_residual' in self.config and self.config['use_residual']:
            return dnn(input_emb) + input_emb
        else:    
            return dnn(input_emb)
        
    def _renew_emb_table(self):
        if 'cancel_renew' in self.config and self.config['cancel_renew']:
            # do nothing
            return
        
        with torch.no_grad():
            # if self.config['renew_by_loading_best']:
            if self.config['renew_by_loading_best'] and (self.total_prop_times >= self.config['K']):
                print("## renew_emb_table_by_loading_best")
                self.emb_table = torch.load(
                    osp.join(self.config['results_root'], 'out_emb_table.pt'),
                    map_location=self.emb_table_device
                )
            else:
                print("## renew_emb_table")
                if self.config['use_two_dnn']:
                    self._infer_dnn_output_emb(
                        self.user_dnn,
                        input_table=self.emb_table[:self.num_users],
                        output_table=self.emb_table[:self.num_users]
                    )
                    self._infer_dnn_output_emb(
                        self.item_dnn,
                        input_table=self.emb_table[self.num_users:],
                        output_table=self.emb_table[self.num_users:]
                    )
                else:
                    self._infer_dnn_output_emb(
                        self.dnn,
                        input_table=self.emb_table,
                        output_table=self.emb_table
                    )
            self._print_emb_info(self.emb_table, 'emb_table')
    
    def get_L2_regularization_loss(self, emb):
        loss_reg = 1/2 * ((emb**2).sum()) / len(emb)
        return loss_reg
    
    def get_augmented_pos(self, pos):
        if self.dataset_type == 'user-item':
            pos -= self.num_users
            
        pos_aug = self.ii_topk_neighbors[pos].flatten()
        # ii_scores = self.ii_topk_similarity_scores[pos]
        
        if self.dataset_type == 'user-item':
            pos_aug += self.num_users
        
        return pos_aug
    
    def get_rank_loss(self, src_emb, pos_emb, neg_emb):
        loss_fn_type = self.config['loss_fn']
        if loss_fn_type == 'bpr_loss':
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            loss = bpr_loss(pos_score, neg_score)
        
        elif loss_fn_type == 'bce_loss':
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            loss = bce_loss(pos_score, neg_score, 
                            self.config['neg_weight'], reduction='mean')
        
        elif loss_fn_type == 'cosine_contrastive_loss':
            loss = cosine_contrastive_loss(
                src_emb, pos_emb, neg_emb,
                self.config['margin'], self.config['neg_weight']
            )
        else:
            assert 0
        
        return loss
    
    def __call__(self, batch_data):
        return self.forward(batch_data)
        
    def forward(self, batch_data):
        src, pos, neg = batch_data
        emb_for_reg_list = []  # embeddings that need to calculate L2 regularization loss
        
        # calculate main loss
        if self.config['use_two_dnn']:
            src_emb = self.get_output_emb(src, self.user_dnn)
            pos_emb = self.get_output_emb(pos, self.item_dnn)
            neg_emb = self.get_output_emb(neg, self.item_dnn)
        else:
            src_emb = self.get_output_emb(src, self.dnn)
            pos_emb = self.get_output_emb(pos, self.dnn)
            neg_emb = self.get_output_emb(neg, self.dnn)
        emb_for_reg_list.extend([src_emb, pos_emb, 
                                 neg_emb.reshape(-1, src_emb.shape[-1])])
        
        loss = self.get_rank_loss(src_emb, pos_emb, neg_emb)
        
        # calculate loss from augmented training samples
        if 'gamma' in self.config and self.config['gamma'] > 0:
            src_aug = src.repeat_interleave(self.config['topk'])
            pos_aug = self.get_augmented_pos(pos)
            neg_aug = self.data['train_dl'].get_neg_samples(src_aug)
            
            src_aug_emb = src_emb.repeat_interleave(self.config['topk'], dim=0)
            if self.config['use_two_dnn']:
                pos_aug_emb = self.get_output_emb(pos_aug, self.item_dnn)
                neg_aug_emb = self.get_output_emb(neg_aug, self.item_dnn)
            else:
                pos_aug_emb = self.get_output_emb(pos_aug, self.dnn)
                neg_aug_emb = self.get_output_emb(neg_aug, self.dnn)    
            emb_for_reg_list.extend([pos_aug_emb, 
                                     neg_aug_emb.reshape(-1, src_emb.shape[-1])])
            
            loss_aug = self.get_rank_loss(src_aug_emb, pos_aug_emb, neg_aug_emb)
            loss += self.config['gamma'] * loss_aug
        
        rw = self.config['l2_reg_weight']
        if rw > 0:
            loss_reg = self.get_L2_regularization_loss(torch.cat(emb_for_reg_list))
            loss += rw * loss_reg
        
        return loss
    
    def prepare_for_train(self):
        if self.config['use_two_dnn']:
            self.user_dnn.train()
            self.item_dnn.train()
        else:
            self.dnn.train()
        
        epoch = self.data['epoch']
        
        if 'max_prop_times' in self.config:
            if self.total_prop_times >= self.config['max_prop_times']:
                # do nothing
                return
        
        if self.total_prop_times < self.config['K']:
            if not (epoch % self.config['renew_and_prop_freq']) and epoch != 0:
                self._renew_emb_table()
                self._do_propagation()
                self.total_prop_times += 1
                self.epoch_last_prop = epoch
        else:
            if 'not_eval' in self.config and self.config['not_eval']:
                # do nothing
                pass
            else:
                if (epoch - self.data['epoch_best']) > self.config['endure'] and \
                (epoch - self.epoch_last_prop) > self.config['endure']:
                    self._renew_emb_table()
                    self._do_propagation()
                    self.total_prop_times += 1
                    self.epoch_last_prop = epoch
    
    def prepare_for_eval(self):
        if self.config['use_two_dnn']:
            self.user_dnn.eval()
            self.item_dnn.eval()
        else:
            self.dnn.eval()
        
        self.out_emb_table = torch.empty(self.emb_table.shape, dtype=torch.float32).to(
            self.emb_table_device
        )
        if self.config['use_two_dnn']:
            self._infer_dnn_output_emb(
                self.user_dnn,
                input_table=self.emb_table[:self.num_users],
                output_table=self.out_emb_table[:self.num_users]
            )
            self._infer_dnn_output_emb(
                self.item_dnn,
                input_table=self.emb_table[self.num_users:],
                output_table=self.out_emb_table[self.num_users:]
            )
        else:
            self._infer_dnn_output_emb(
                self.dnn,
                input_table=self.emb_table,
                output_table=self.out_emb_table
            )
        
        if 'zero_degree_zero_emb' in self.config and self.config['zero_degree_zero_emb']:
            self.out_emb_table[self.zero_degree_nodes] = 0
        
        self._print_emb_info(self.out_emb_table, 'out_emb_table')
        
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table

    def save(self, root, file_out_emb_table=None):
        if file_out_emb_table is None:
            file_out_emb_table = "out_emb_table.pt"
        torch.save(self.out_emb_table, osp.join(root, file_out_emb_table))
        
        self.export_embedding_as_npy(root)

    def export_embedding_as_npy(self, root):
        is_user_item = True if self.dataset_type == 'user-item' else False
        out_embeddings = self.out_emb_table.cpu().detach().numpy()
        if not is_user_item:
            file_name = 'node_embedding.npy' 
            np.save(osp.join(root, file_name), out_embeddings)
        else:
            file_name_user = 'user_embedding.npy'
            file_name_item = 'item_embedding.npy'
            num_users = self.info['num_users']
            np.save(osp.join(root, file_name_user), out_embeddings[:num_users])
            np.save(osp.join(root, file_name_item), out_embeddings[num_users:])