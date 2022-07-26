from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.EmbeddingTable.EmbeddingMultiplexer import EmbeddingMultiplexer
from model.module import dot_product, bpr_loss
from utils import xconfig

import torch
from tqdm import tqdm
import os.path as osp


class BaseGNNModel(BaseEmbeddingModel):

    def __init__(self, config, data):
        super().__init__(config, data)
        self.root_config = config
        self.config_field_name = ''
        
        self.local_config = self.root_config[self.config_field_name]
        
        self.device = self.root_config['device']
        
        # self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        self.base_emb_table = EmbeddingMultiplexer(
            config=config, 
            config_field_name=xconfig.join(self.config_field_name, 'EmbeddingMultiplexer'), 
            data=data
        )
        
        # init by subclass
        self.param_list = None
        self.gnn = None
        
    def __call__(self, batch_data):
        return self.forward(batch_data)

    def forward(self, batch_data):
        batch_nids, local_idx, input_nids, output_nids, blocks = batch_data
    
        blocks = [block.to(self.device) for block in blocks]
        
        output_embs = self.gnn(
            blocks, self.base_emb_table(input_nids.to(self.device))
        )
        
        output_embs = output_embs[local_idx].view(3, -1, self.base_emb_table.shape[-1])
        
        src_emb = output_embs[0, :, :]
        pos_emb = output_embs[1, :, :]
        neg_emb = output_embs[2, :, :]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss = bpr_loss(pos_score, neg_score)

        rw = self.local_config['l2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(output_embs)) * (output_embs**2).sum()
            loss += rw * L2_reg_loss
        
        return loss
    
    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        node_collator = self.data['node_collator']
        self.out_emb_table = torch.empty(self.base_emb_table.shape, dtype=torch.float32).to(self.device)
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.num_nodes), 
                                         batch_size=8192,
                                         collate_fn=node_collator.collate)
        
        for input_nids, output_nids, blocks in tqdm(dl, desc="get all gnn output embs"):
            blocks = [block.to(self.device) for block in blocks]
            output_embs = self.gnn(
                blocks, self.base_emb_table(input_nids.to(self.device))
            )
            self.out_emb_table[output_nids] = output_embs
        
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
    
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
