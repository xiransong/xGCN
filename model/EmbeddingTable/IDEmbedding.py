import torch

from utils import io


class IDEmbedding:
    
    def __init__(self, config, config_field_name, data):
        self.root_config = config
        self.config_field_name = config_field_name
        self.data = data
        
        self.local_config = self.root_config[self.config_field_name]
        self.info = self.data['info']
        
        self.num_nodes = self.info['num_nodes']
        self.device = self.local_config['device']
        
        if 'from_pretrained' in self.local_config and self.local_config['from_pretrained']:
            file = self.local_config['file_pretrained_emb']
            if file[-3:] == '.pt':
                emb_table = torch.load(file)
            else:
                emb_table = io.load_pickle(file)
                emb_table = torch.FloatTensor(emb_table)
            self.emb_dim = emb_table.shape[-1]
        else:
            self.emb_dim = self.local_config['emd_dim']
            emb_table = torch.FloatTensor(size=(self.num_nodes, self.emb_dim))
            torch.nn.init.normal_(emb_table, 
                                  mean=0.0, std=self.local_config['emb_init_std'])
        self._shape = (self.num_nodes, self.emb_dim)

        use_sparse = bool('use_sparse' in self.local_config and self.local_config['use_sparse'])
        freeze = bool('freeze' in self.local_config and self.local_config['freeze'])
        
        self.emb_table = torch.nn.Embedding.from_pretrained(
            emb_table, freeze=freeze,
            sparse=use_sparse
        ).to(self.device)
        
        self.param_list = {
            'SparseAdam': [],
            'Adam': []
        }
        if not freeze:
            if use_sparse:
                self.param_list['SparseAdam'].append(
                    {'params': list(self.emb_table.parameters()), 
                    'lr': self.local_config['lr']}
                )
            else:
                self.param_list['Adam'].append(
                    {'params': self.emb_table.parameters(), 
                    'lr': self.local_config['lr']}
                )
    
    def get_param_list(self):
        return self.param_list
    
    @property
    def shape(self):
        return self._shape
    
    def __call__(self, nids):
        return self.forward(nids)
    
    def forward(self, nids):
        # get node ID embedding
        return self.emb_table(nids.to(self.device))

    def infer_all_emb(self):
        return self.emb_table.weight
