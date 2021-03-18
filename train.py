#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# In[2]:


import sys
sys.argv = sys.argv[:1]
import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Video Re-loc")

parser.add_argument('--inter_op_parallelism_threads', default=0, type=int,
                    help='number of threads')
parser.add_argument('--intra_op_parallelism_threads', default=0, type=int,
                    help='number of threads')                    
parser.add_argument('--max_length', default=300, type=int,
                    help='max length')
parser.add_argument('--feat_dim', default=500, type=int,
                    help='feature dim')
parser.add_argument('--keep_prob', default=0.6, type=float,
                    help='keep prob')                    
parser.add_argument('--mem_dim', default=128, type=int,
                    help='hidden state dim')
parser.add_argument('--att_dim', default=128, type=int,
                    help='attention dim')    
parser.add_argument('--job_dir', default='saving', type=str,
                    help='job_dir')    
parser.add_argument('--data_dir', default='data', type=str,
                    help='dir')               
parser.add_argument('--num_gpus', default=0, type=int,
                    help='number of gpus')
parser.add_argument('--bucket_span', default=30, type=int,
                    help='bucket_span')                    
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch_size')
parser.add_argument('--max_steps', default=1000, type=int,
                    help='max_steps')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay')                    
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learning_rate')
parser.add_argument('--max_gradient_norm', default=5.0, type=float,
                    help='max_gradient_norm')    
parser.add_argument('--save_summary_steps', default=10, type=int,
                    help='save_summary_steps')  
parser.add_argument('--save_checkpoint_steps', default=100, type=int,
                    help='save_checkpoint_steps')  
global args
args = parser.parse_args()


# In[3]:

class Dataset(data.Dataset):
    def __init__(self, args, subset):
        self.is_training = subset == 'train'
        with open(os.path.join('data', subset + '.json'), 'r') as f:
            data = json.load(f)
        self.videos = [[] for _ in range(4)]
        for i in data:
            self.videos[0].append(i['id'])
            self.videos[1].append(i['groundtruth'])
            self.videos[2].append(i['label'])
            self.videos[3].append(i['location'])
        for i in range(1,4):
            self.videos[i] = torch.tensor(self.videos[i])

        self.dataset = self.videos

    def __getitem__(self, index):
        def map_fn(func,inputs,extra=None):
            outputs = []
            length = len(inputs[0])
            for i in range(length):
                inp_func = []
                for x in inputs:
                    inp_func.append(x[i])
                d = np.copy(inp_func)
                if extra:
                    inp_func.extend(extra)
                #print(inp_func)
                samp = func(inp_func)
                if samp == None:
                    continue
                else:
                    outputs.append(samp)
            return outputs

        def map_fn2(func,inputs,extra=None):
            outputs = []
            length = len(inputs[0])
            for x in inputs:
                inp_func = []
                for i in range(length):
                    inp_func.append(x[i])
                if extra:
                    inp_func.extend(extra)
                outputs.append(func(inp_func))
            return outputs

        def sample(inputs):
            query_id, query_gt, query_label, query_loc, all_ids, all_gts, all_labels, all_locs, is_training = inputs

            same =(all_labels == query_label)
            longer = query_gt[1] - query_gt[0] < all_locs[:,1] - all_locs[:,0]
            same = torch.logical_and(same,longer)
            same = torch.where(same)[0]
            num = same.shape[0]
            if num == 0:
                return
            idx = np.random.choice(same)
            chosen_id = all_ids[idx]
            chosen_gt = all_gts[idx]
            chosen_loc = all_locs[idx]
            if is_training:
                off_st = torch.randint(chosen_gt[0] + 1,[])
                maxval = chosen_loc[1] - chosen_loc[0] - chosen_gt[1] + 1
                off_en = torch.randint(maxval,[])
                use_off = torch.rand([])
                off_st = off_st if use_off < 0.9 else torch.tensor(0)
                off_en = off_en if use_off < 0.9 else torch.tensor(0)
                off_gt = torch.stack([-off_st,-off_st])
                off_loc = torch.stack([off_st, -off_en])
                chosen_gt += off_gt
                chosen_loc += off_loc
            return query_id, query_gt + query_loc[0], chosen_id, chosen_gt, chosen_loc


        def get_data(inputs):
            v1, t1, v2, t2, l2,data_dir = inputs
            """Read the video features."""
            feat1 = np.load('%s/feat/v_%s.npy' % (data_dir, v1))
            feat2 = np.load('%s/feat/v_%s.npy' % (data_dir, v2))
            len1 = t1[1] - t1[0]
            len2 = l2[1] - l2[0]
            ret1 = feat1[t1[0]:t1[1]]
            ret2 = feat2[l2[0]:l2[1]] 
            assert len1 == ret1.shape[0]
            assert len2 == ret2.shape[0]
            assert (t2 >= 0).all() and (t2 <= len2).all()
            return ret1, len1, ret2, len2, t2

        def get_bucket_span(ret):
            gt = [[]for i in range(11)]
            #print(ret)
            for i in ret:
                #print(i)
                l = i[-1][1]-i[-1][0]
                l = l // args.bucket_span
                gt[l].append(i)
            return gt
        
        def random_choice(gt):
            l = []
            for x in gt:
                l.append(len(x))
            total=np.sum(l)
            for i in range(len(l)):
                l[i]=l[i]/total
            choice_bucket = np.random.choice(gt,p=l)
            choice_batch = np.random.choice(len(choice_bucket),replace=True,size=args.batch_size)
            ret = []
            for index in choice_batch:
                ret.append(choice_bucket[index])
            return ret
        
        ret = map_fn(sample,self.dataset,[*self.videos,self.is_training])
        gt = get_bucket_span(ret)
        ret = random_choice(gt)
        ret = map_fn2(get_data,ret,[args.data_dir])
        
        return ret
    def __len__(self):
        return args.max_steps+1

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
        self.Match1=MatchCellWrapper(self.fwd, self.mem_dim,self.att_dim)
        self.Match2=MatchCellWrapper(self.bwd, self.mem_dim,self.att_dim)
        self.pointer = nn.LSTMCell(self.att_dim*2,self.att_dim)
        self.fc = nn.Linear(self.att_dim,4)
        self.maxlen=300
        
    def flipBatch(self,data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        ret = data.clone()
        for i in range(data.shape[0]):
            ret[i,:lengths[i]] = data[i,:lengths[i]].flip(dims=[0])
        return ret    
    
    def dynamic_rnn(self, model, inputs, seq_len):
        max_time = inputs.shape[1]
        batch = inputs.shape[0]
        hx = torch.zeros(batch, model.hidden_size)
        cx = torch.zeros(batch, model.hidden_size)
        none = 0
        for i in range(max_time):
            hx,cx = model(inputs[:,i,:],(hx,cx))
            for k in range(batch):
                hx[k][int(seq_len[k]):]=0
                cx[k][int(seq_len[k]):]=0
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
            hx,cx = model(inputs1, inputs2[:,i,:],(hx,cx),seq_len)
            for k in range(batch):
                hx[k][int(seq_len[k]):]=0
                cx[k][int(seq_len[k]):]=0
            if(none == 0):
                ret = torch.unsqueeze(hx,axis=1)
                none = 1
            else:
                ret = torch.cat((ret,torch.unsqueeze(hx,axis=1)),axis=1)
        return ret    
    
    def forward(self,features,labels):
        query,len_q,ref,len_r = features
        query = torch.autograd.Variable(query)
        ref = torch.autograd.Variable(ref)
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
        logits = self.fc(point_out)
         
        return logits
    
    def predict(self,logits,length):#len_r
        
        def map_fn(func,inputs):
            outputs = []
            length = inputs[0].shape[0]
            for i in range(length):
                inp_func = []
                for x in inputs:
                    inp_func.append(x[i])
                outputs.append(func(inp_func))
            return torch.stack(outputs)  
        
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
                    score = self.flipBatch(score, torch.zeros([it]).int() + length)
                    score = self.flipBatch(score, length - torch.range(0,it-1).int())
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
        
        return map_fn(map_body,[logits,length])

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
        
        labels = all_labels
        idx = torch.stack([torch.range(0,self.batch_size-1), labels[:, 0]], axis=1)
        label_st = scatter_nd(idx, torch.ones(self.batch_size),[self.batch_size, self.maxlen])
        idx = torch.stack([torch.range(0,self.batch_size-1), labels[:, 1] - 1], axis=1)
        label_en = scatter_nd(idx, torch.ones(self.batch_size), [self.batch_size, self.maxlen])
        inside_t = sequence_mask(labels[:, 1] - labels[:, 0], self.maxlen)
        inside_t = self.flipBatch(inside_t, labels[:, 1])
        outside = torch.logical_not(inside_t)
        inside_t = inside_t.float()
        outside = outside.float()
        label = torch.stack([label_st, label_en, inside_t, outside], axis=2)

        # Eq. (10)
        heavy = label[:, :, :2].sum(-1) > 0.9
        heavy = heavy * 9 + 1
        label = label / torch.sum(label, axis=2, keepdims=True)
        
        loss = torch.zeros(logits.shape[0],logits.shape[1])
        softmax = torch.softmax(logits,dim=-1)
        for k in range(logits.shape[2]):
            loss += -label[:,:,k]*torch.log(softmax[:,:,k])
        loss *= heavy
        mask = sequence_mask(torch.tensor(len_r), self.maxlen)
        loss = loss[mask]
        loss = torch.mean(loss)
        #model_params = optimizer.param_groups
        #weights = [i for i in model_params if 'bias' not in i.name]
        #loss += params.weight_decay * torch.add_n([torch.nn.MSELoss(v) for v in weights])
        
        return loss
    
    def evaluate(self,predictions, label):
        def get_iou(pred,label):
            pred_l, pred_r = torch.unbind(pred,1)
            for i in range(2, len(pred.shape)):
                label = label.unsqueeze(i)
            label_l, label_r = torch.unbind(label,1)
            inter_l = torch.max(pred_l, label_l)
            inter_r = torch.min(pred_r, label_r)
            inter = torch.max((inter_r - inter_l).float(), torch.zeros(inter_l.shape))
            union = pred_r - pred_l + label_r - label_l - inter
            return inter/union
        def get_eval_metric(iou):
            th = np.arange(0.1, 1.0, 0.1)
            vals = [(iou > i).float() for i in th]
            vals = torch.stack(vals[4:]).T
            vals = torch.mean(vals)
            return vals
        
        iou = get_iou(predictions,label)
        metrics = get_eval_metric(iou)
        return metrics


# In[15]:


class MatchCellWrapper(nn.Module):
    def __init__(self, cell, input_size, output_size, reuse=None):
        super(MatchCellWrapper, self).__init__()
        self._cell = cell
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

    def forward(self, _hq, inputs, state, length):
        hq = self.fcq(_hq)
        _length = length
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
        
        hq = self.map_fn(body,[g, _hq, _length])
        gate_hq = self.fchq(inputs)
        gate_in = self.fcin(hq)
        gate_hq = self.sigmoid(gate_hq)
        gate_in = self.sigmoid(gate_in)
        hq1 = hq* gate_hq
        inputs1 = inputs * gate_in
        
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
        
        inputs = bilinear(inputs1, hq1)
        h,c = self._cell(inputs1, (h,c))        
        return (h,c)


    
def collate_fn(batch):
    query = [torch.Tensor(item[0]) for item in batch[0]]
    query = pad_sequence(query,batch_first=True)
    len_q = [item[1] for item in batch[0]]
    ref = [torch.Tensor(item[2]) for item in batch[0]]
    ref = pad_sequence(ref,batch_first=True)
    len_r=[item[3] for item in batch[0]]
    label =[torch.tensor(item[-1]) for item in batch[0]]
    label = torch.stack(label)
    return query,len_q,ref,len_r,label

# In[13]:

data= DataLoader(Dataset(args,subset="train"),shuffle=True,collate_fn=collate_fn,pin_memory=True)

model = model_fn(args)
#model.load_state_dict(torch.load("saving/model10.pt"))
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
   
    #model.load_state_dict(torch.load("saving/model10.pt"))
for (i,item) in enumerate(data):
    s = time.time()
    query,len_q,ref,len_r,labels = item
    features = [query,len_q,ref,len_r]
    out = model(features,labels)
    loss = model.get_loss(out,labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t = time.time()
    print(i, query.shape[1], ref.shape[1], loss, t-s)
    if i%100 == 0:
      torch.save(model.state_dict(), "saving/modelul"+str(i)+".pt")
