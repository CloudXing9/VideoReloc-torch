import torch
import sys
import torch.nn as nn
import numpy as np
from opts import parser
global args = parser.parse_args()

class model_fn(nn.Module):
    def __init__(self,args):
        super(model_fn, self).__init__()
        #query, len_q, ref, len_r = features
        self.batch_size = args.batch_size
        self.features = args.feat_dim
        self.seq_length = args.max_length
        self.mem_dim = args.mem_dim
        self.att_dim = args.att_dim
        self.keep_prob = args.keep_prob
        
        self.cell = nn.LSTMCell(self.features,self.mem_dim)
        self.dropout = nn.Dropout(self.keep_prob)
        self.fwd = nn.LSTMCell(self.mem_dim,self.att_dim)
        self.bwd = nn.LSTMCell(self.mem_dim,self.att_dim)
        self.Match1=MatchCellWrapper(self.fwd, self.mem_dim,self.att_dim,len_q)
        self.Match2=MatchCellWrapper(self.bwd, self.mem_dim,self.att_dim,len_q)
        self.pointer = nn.LSTMCell(self.att_dim*2,self.att_dim)
        self.fc = nn.Linear(self.att_dim,4)
        self.maxlen=300
        
    def flipBatch(self,data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        for i in range(data.shape[0]):
            data[i,:lengths[i]] = data[i,:lengths[i]].flip(dims=[0])
        return data    
    
    def dynamic_rnn(self, model, inputs, seq_len):
        max_time = inputs.shape[1]
        batch = inputs.shape[0]
        hx = torch.zeros(batch, model.hidden_size)
        cx = torch.zeros(batch, model.hidden_size)
        none = 0
        for i in range(max_time):
            hx,cx = model(inputs[:,i,:],(hx,cx))
            for k in range(batch):
                hx[k][seq_len[k]:]=0
                cx[k][seq_len[k]:]=0
            if(none == 0):
                ret = torch.unsqueeze(hx,axis=1)
                none = 1
            else:
                ret = torch.cat((ret,torch.unsqueeze(hx,axis=1)),axis=1)
        return ret
    
    def dynamic_rnn_match(self, model,inputs1, inputs2, seq_len):
        max_time = inputs2.shape[1]
        batch = inputs2.shape[0]
        hx = torch.zeros(batch, model._attn_vec_size)
        cx = torch.zeros(batch, model._attn_vec_size)
        none = 0
        for i in range(max_time):
            hx,cx = model(inputs1, inputs2[:,i,:],(hx,cx))
            for k in range(batch):
                hx[k][seq_len[k]:]=0
                cx[k][seq_len[k]:]=0
            if(none == 0):
                ret = torch.unsqueeze(hx,axis=1)
                none = 1
            else:
                ret = torch.cat((ret,torch.unsqueeze(hx,axis=1)),axis=1)
        return ret    
    
    def forward(self,features,labels):
        query,len_q,ref,len_r = features
        out1 = self.dynamic_rnn(self.cell, query, len_q)
        out2 = self.dynamic_rnn(self.cell, ref, len_r)
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)
        
        forward_out = self.dynamic_rnn_match(self.Match1,out1,out2,len_q)
        out2_reverse = self.flipBatch(out2,len_r)
        backward_out = self.dynamic_rnn_match(self.Match2,out1,out2_reverse,len_q)
        backward_out = self.flipBatch(backward_out,len_r)
        h = torch.cat([forward_out,backward_out],axis=2)
        h = self.dropout(h)
        self.maxlen = h.shape[1]
        point_out = self.dynamic_rnn(self.pointer,h,len_r)
        logits = self.fc(pointer)
         
        return logits
    
    def predict(self,logits,length):
        def map_body(x):
            logits = x[0]
            length = x[1]
            logits = logits[:length]
            prob = torch.nn.functional.log_softmax(logits, dim=1)
            prob = prob.T

            initial_it = 0
            initial_idx_ta = []
            initial_val_ta = []

            def cond(inp):
                it = inp[0]
                return it < min(length,64)

            def while_body(x):
                it, idx_ta, val_ta=x
                if it == 0:
                        total = prob[:2].sum(0)
                else:
                        total = prob[0, :-it] + prob[1, it:]

                def get_inside():
                    score = prob[2, None, :].repeat([it, 1])
                    score = flipBatch(score, torch.zeros([it]).int() + length)
                    score = flipBatch(score, length - torch.range(0,it-1).int())
                    score = score[:, :-it]
                    score = score.mean(0)
                    return score
                if it ==0:
                    ave = prob[2]
                else:
                    ave = get_inside()

                total += ave
                idx = torch.argmax(total)
                idx_ta.append(idx)
                val_ta.append(total[idx])
                it += 1
                return it, idx_ta, val_ta

            res = [initial_it, initial_idx_ta, initial_val_ta]
            while(cond(res)):
                res = while_body(res)
            final_idx = torch.stack(res[1])
            final_val = torch.stack(res[2])
            idx = torch.argmax(final_val)
            pred = torch.stack([final_idx[idx], final_idx[idx] + idx + 1])
            return pred
        map_fn(map_body,[logits,length])

    def get_loss(self,logits,all_labels):
        #loss
        def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
            if maxlen is None:
                 maxlen = lengths.max()
            row_vector = torch.arange(0, maxlen, 1)
            matrix = torch.unsqueeze(lengths, dim=-1)
            mask = row_vector < matrix
            mask.type(dtype)
            return mask
        
        def scatter_nd(ind,update,shape):
            scatter = torch.zeros(shape)
            for i in range(len(update)):
                scatter[ind[i][0].long(),ind[i][1].long()]=update[i]
            return scatter
        
        labels = torch.tensor(all_labels)
        idx = torch.stack([torch.range(0,self.batch_size-1), labels[:, 0]], axis=1)
        label_st = scatter_nd(idx, torch.ones(self.batch_size),[self.batch_size, self.maxlen])
        idx = torch.stack([torch.range(0,self.batch_size-1), labels[:, 1] - 1], axis=1)
        label_en = scatter_nd(idx, torch.ones(self.batch_size), [self.batch_size, self.maxlen])
        inside_t = sequence_mask(labels[:, 1] - labels[:, 0], self.maxlen)
        inside_t = flipBatch(inside_t, labels[:, 1])
        outside = torch.logical_not(inside_t)
        inside_t = inside_t.float()
        outside = outside.float()
        label = torch.stack([label_st, label_en, inside_t, outside], axis=2)

        # Eq. (10)
        heavy = label[:, :, :2].sum(-1) > 0.9
        heavy = torch.tensor(heavy) * 9 + 1
        label = label / torch.sum(label, axis=2, keepdims=True)
        '''
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(label,logits)
        loss *= heavy
        mask = sequence_mask(len_r, self.maxlen)
        loss = loss[mask]
        loss = torch.mean(loss)
        model_params = optimizer.param_groups
        weights = [i for i in model_params if 'bias' not in i.name]
        loss += params.weight_decay * torch.add_n([torch.nn.MSELoss(v) for v in weights])
        '''
        
class MatchCellWrapper(nn.Module):
    def __init__(self, cell, input_size, output_size, length, reuse=None):
        super(MatchCellWrapper, self).__init__()
        self._cell = cell
        self._length = length
        self.features = input_size
        self._attn_vec_size = output_size
        self.fcq = nn.Linear(self.features,self._attn_vec_size,bias=None)
        self.fcp = nn.Linear(2*self._attn_vec_size,self._attn_vec_size)
        self.tanh = nn.Tanh()
        self.fcg = nn.Linear(self._attn_vec_size,1)
        self.softmax = nn.Softmax()
        
        self.fchq = nn.Linear(self.features,self._attn_vec_size)
        self.fcin = nn.Linear(self.features,self._attn_vec_size)

        self.fcx = nn.Linear(self._attn_vec_size,self._attn_vec_size*8)
        self.fcy = nn.Linear(self._attn_vec_size,self._attn_vec_size*8)
        self.sigmoid = nn.Sigmoid()
    @property
    def output_size(self):
        return self._attn_vec_size

    @property
    def state_size(self):
        return [self._attn_vec_size, self._attn_vec_size]
    
    def map_fn(self,func,inputs):
        outputs = []
        length = inputs[0].shape[0]
        for i in range(length):
            inp_func = []
            for x in inputs:
                inp_func.append(x[i])
            outputs.append(func(inp_func))
        return torch.stack(outputs)

    def forward(self, _hq, inputs, state):
        hq = self.fcq(_hq)
        h,c = state
        concat = torch.cat((inputs, h), axis=1)
        hp = self.fcp(concat)
        hp = torch.unsqueeze(hp, 1)
        g = self.tanh(hq+hp)
        g = self.fcg(g)
        g = torch.squeeze(g, 2)
            
        def body(x):
            alpha = x[0]
            hq = x[1]
            length = x[2]
            alpha = alpha[:length]
            hq = hq[:length]
            alpha = self.softmax(alpha)
            hq = (hq * alpha[:, None]).sum(0)
            return hq
        
        hq = self.map_fn(body,[g, _hq, self._length])
        gate_hq = self.fchq(inputs)
        gate_in = self.fcin(hq)
        gate_hq = self.sigmoid(gate_hq)
        gate_in = self.sigmoid(gate_in)
        hq *= gate_hq
        inputs *= gate_in
        
        def bilinear(x, y, num_outputs=None, k=8):
            _, input_dim = x.shape
            if num_outputs is None:
                num_outputs = input_dim
            x = self.fcx(x)
            y = self.fcy(y)
            x = torch.reshape(x, [-1, num_outputs, k])
            y = torch.reshape(y, [-1, num_outputs, k])
            bi = (x * y).sum(-1)
            return bi        
        
        inputs = bilinear(inputs, hq)
        h,c = self._cell(inputs, (h,c))        
        return (h,c)



