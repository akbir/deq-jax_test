import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../DEQModel/")
from modules.deq import *

__author__ = "shaojieb"


class TransformerDEQModule(DEQModule):

    """ See DEQModule class for documentation """

    def __init__(self, func, func_copy) -> object:
        super(TransformerDEQModule, self).__init__(func, func_copy)
        
    def _solve_by_subseq(self, z1s, us, z0, pos_emb, threshold, train_step, subseq_len=100):
        z1s_out = torch.zeros_like(z1s)
        with torch.no_grad():
            z0_temp = z0
            for t in range(0, seq_len, subseq_len):
                prev_len = z0_temp.size(2)
                z1ss = z1s[:,:,t:t+subseq_len]     # The subsequence of z1s processed here
                uss = us[:,:,(t+pad_len-prev_len):(t+pad_len+subseq_len)]
                pos_embss = pos_emb[:,:,-uss.size(2):]     # The last (prev_len + subseq_len) positions

                # Invoke DEQ forward module. args = [pos_embss, threshold, train_step]
                res = RootFind.apply(self.func, z1ss, uss, z0_temp, pos_embss, threshold, train_step)
                z0_temp = res
                z1s_out[:,:,t:t+subseq_len] = res
        
        z1s = z1s_out
        z1s_out = torch.zeros_like(z1s)
        z0_temp = z0
        for t in range(0, seq_len, subseq_len):
            prev_len = z0_temp.size(2)
            z1ss = z1s[:,:,t:t+subseq_len]     # The subsequence of z1s processed here
            uss = us[:,:,(t+pad_len-prev_len):(t+pad_len+subseq_len)]
            pos_embss = pos_emb[:,:,-uss.size(2):]     # The last (prev_len + subseq_len) positions

            new_z1ss = RootFind.f(self.func, z1ss, uss, z0_temp, pos_embss, threshold, train_step)
            res = self.Backward.apply(self.func_copy, z1s_out, us, z0, pos_emb, threshold, train_step)
            z0_temp = res
            z1s_out[:,:,t:t+subseq_len] = res
        return z1s_out

    def forward(self, z1s, us, z0, **kwargs):
        bsz, d_model, seq_len = z1s.size()
        pad_len = z0.size(2)
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        pos_emb = kwargs.get('pos_emb', None)
        subseq_len = kwargs.get('subseq_len', seq_len)

        if us is None or pos_emb is None:
            raise ValueError("Input injection and positional encodings are required.")

        # Use this line for longer sequences: 
        #     self._solve_by_subseq(z1s, us, z0, pos_emb, threshold, train_step, subseq_len=subseq_len)

        # Use these lines for shorter sequences:
        z1s_out = RootFind.apply(self.func, z1s, us, z0, pos_emb, threshold, train_step)
        if self.training:
            z1s_out = RootFind.f(self.func, z1s_out, us, z0, pos_emb, threshold, train_step)
            z1s_out = self.Backward.apply(self.func_copy, z1s_out, us, z0, pos_emb, threshold, train_step)
        return z1s_out

        