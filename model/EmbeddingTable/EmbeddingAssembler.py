from model.EmbeddingTable.IDEmbedding import IDEmbedding
from utils import xconfig
from utils.utils import merge_list_dict

import torch


class EmbeddingAssembler:
    
    def __init__(self, config, config_field_name, data):
        self.root_config = config
        self.config_field_name = config_field_name
        self.data = data
        
        self.local_config = self.root_config[self.config_field_name]
        self.info = self.data['info']
        
        self.num_nodes = self.info['num_nodes']
        self.output_emb_dim = self.local_config['output_emd_dim']
        self._shape = (self.num_nodes, self.output_emb_dim)
        
        self.device = self.local_config['device']
        
        self.param_list = {
            'SparseAdam': [],
            'Adam': []
        }
        
        # build embedding tables
        ## one learnable embedding for each node
        self.id_emb_table = IDEmbedding(
            config=config, 
            config_field_name=xconfig.join(self.config_field_name, 'id_emb_table'),
            data=data
        )
        self.param_list = merge_list_dict(
            self.param_list, self.id_emb_table.get_param_list()
        )
        
        ## node feature embedding
        self.feat_emb_table_list = []
        if 'feat_emb_tables' in self.local_config:
            for table_name in self.local_config['feat_emb_tables']:
                table = IDEmbedding(
                    config=config, 
                    config_field_name=xconfig.join(
                        self.config_field_name, 'feat_emb_tables', table_name
                    ),
                    data=data
                )
                self.feat_emb_table_list.append(table)
                self.param_list = merge_list_dict(self.param_list, table.get_param_list())
        self.num_feat = len(self.feat_emb_table_list)
        
        #  MLP for projecting feature embedding
        if self.num_feat == 0:
            self.feat_mlp = None
        else:
            self.feat_mlp = torch.nn.Sequential(
                *eval(self.local_config['feat_mlp']['arch'])
            ).to(self.device)
            self.param_list['Adam'].append({
                'params': self.feat_mlp.parameters(),
                'lr': self.local_config['feat_mlp']['lr']
            })
        
    def get_param_list(self):
        return self.param_list
        
    @property
    def shape(self):
        return self._shape
    
    def __call__(self, nids):
        return self.forward(nids)
    
    def forward(self, nids):
        # get node ID embedding
        id_emb = self.id_emb_table(nids)
        
        if self.num_feat == 0:
            return id_emb
        
        # get feature embedding
        ## get embedding for each feature
        feat_emb_list = []
        for table in self.feat_emb_table_list:
            feat_emb_list.append(table(nids))
        
        ## concat all the feature embeddings
        feat_emb = torch.cat(feat_emb_list, dim=-1)
        
        ## project feat_emb to the same length as the node ID embedding
        feat_emb = self.feat_mlp(feat_emb)
        
        # merge id embedding and feature embedding
        emb = id_emb + feat_emb
        
        return emb
    
    def infer_all_emb(self):
        if self.num_feat == 0:
            return self.id_emb_table.infer_all_emb()
        
        X = torch.empty(self.shape,
                        dtype=torch.float32, device=self.device)
        dl = torch.utils.data.DataLoader(torch.arange(self.num_nodes), batch_size=8192)
        for nids in dl:
            nids = nids.to(self.device)
            X[nids] = self.forward(nids)
        return X
