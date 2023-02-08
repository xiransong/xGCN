import torch


class OptiManager:
    
    def __init__(self, param_list):
        self.opt_list = []
        for opt_type in param_list:
            opt = eval('torch.optim.' + opt_type)(
                param_list[opt_type]
            )
            self.opt_list.append(opt)
    
    def zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()
    
    def step(self):
        for opt in self.opt_list:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],
                                           max_norm=1.0, norm_type=2.0)
            opt.step()
