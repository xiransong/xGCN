from utils import xconfig
from utils.utils import merge_list_dict
from model.EmbeddingTable.EmbeddingAssembler import EmbeddingAssembler

import torch


class EmbeddingMultiplexer:
    
    def __init__(self, config, config_field_name, data):
        self.root_config = config
        self.config_field_name = config_field_name
        self.data = data
        
        self.local_config = self.root_config[self.config_field_name]
        
        self.info = data['info']
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        self.num_nodes = self.info['num_nodes']
        self._shape = (self.info['num_nodes'], self.local_config['emb_dim'])
        self.device = config['device']
        
        self.param_list = {}
        
        # build user/item tower
        self.user_tower = EmbeddingAssembler(
            config=config,
            config_field_name=xconfig.join(self.config_field_name, 'user_tower'),
            data=data
        )
        self.param_list = merge_list_dict(
            self.param_list, self.user_tower.get_param_list()
        )
        
        if self.dataset_type == 'user-item':
            self.item_tower = EmbeddingAssembler(
                config=config,
                config_field_name=xconfig.join(self.config_field_name, 'item_tower'),
                data=data
            )
            self.param_list = merge_list_dict(
                self.param_list, self.item_tower.get_param_list()
            )
    
    def get_param_list(self):
        return self.param_list
    
    @property
    def shape(self):
        return self._shape
    
    def _splitter(self, nids):
        mask = nids < self.num_users
        
        user_ids = nids[mask]
        item_ids = nids[~mask] - self.num_users
        
        idx = torch.arange(len(nids)).to(self.device)
        re_align = torch.cat([idx[mask], idx[~mask]])
        
        return user_ids, item_ids, re_align
    
    def __call__(self, nids):
        return self.forward(nids)
    
    def forward(self, nids):
        if self.dataset_type == 'user-item':
            # split flow for different towers
            user_ids, item_ids, re_align = self._splitter(nids)
            
            user_emb = self.user_tower(user_ids)
            item_emb = self.item_tower(item_ids)
            
            # re-align the embedings as the original order as nids
            emb = torch.cat([user_emb, item_emb])[re_align]
        else:
            emb = self.user_tower(nids)
        return emb
    
    def infer_all_emb(self):
        X = torch.empty(self.shape,
                dtype=torch.float32, device=self.device)
        dl = torch.utils.data.DataLoader(torch.arange(self.num_nodes), batch_size=8192)
        for nids in dl:
            nids = nids.to(self.device)
            X[nids] = self.forward(nids)
        return X
