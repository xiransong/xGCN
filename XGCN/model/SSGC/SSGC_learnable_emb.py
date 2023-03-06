from model.BaseGNNModel import BaseGNNModel

import torch
import dgl


class SSGC_learnable_emb_Module(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
        self.mlp = torch.nn.Linear(emb_dim, emb_dim)
        
    def forward(self, blocks, x):
        num_output_nodes = blocks[-1].number_of_dst_nodes()
        x_of_different_layers = [
            x[:num_output_nodes], # embeddings of layer zero
        ]
        for block in blocks:
            block.srcdata['h'] = x
            block.update_all(self.gcn_msg, self.gcn_reduce)
            x = block.dstdata['h']
            x_of_different_layers.append(x[:num_output_nodes])
        x = torch.stack(x_of_different_layers, dim=1).mean(dim=1)
        return self.mlp(x)

    
class SSGC_learnable_emb(BaseGNNModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        
        # add edge_weights to the graph
        g = data['node_collate_graph']  # undirected
        src, dst = g.edges()
        degrees = g.out_degrees()
        d1 = degrees[src]
        d2 = degrees[dst]
        edge_weights = (1 / (d1 * d2)).sqrt()
        g.edata['ew'] = edge_weights
              
        self.gnn = SSGC_learnable_emb_Module(
            emb_dim=self.config['emb_dim'],
        ).to(self.device)
        self.param_list.append({'params': self.gnn.parameters(), 'lr': config['gnn_lr']})
