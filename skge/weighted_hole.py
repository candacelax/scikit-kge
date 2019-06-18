import numpy as np
from itertools import count
from random import shuffle
import torch, timeit
import torch.nn as nn
from torch.optim import Adagrad
from torch.autograd import grad as fn_grad

from skge.torch_util import ccorr, grad_sum_matrix, cconv


_DEF_MAX_EPOCHS = 100
_DEF_LEARNING_RATE = 0.1
# TODO add pairwise

''' TODO '''
class WeightedHolE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WeightedHolE, self).__init__()
        # self.add_hyperparam('rparam', kwargs.pop('rparam', 0.0))

        self.learning_rate = kwargs.get('lr', _DEF_LEARNING_RATE)
        entity_dim, _, relation_dim = args[0]
        embed_dim = args[1]
        self._max_epochs = kwargs.get('max_epochs', _DEF_MAX_EPOCHS)
        
        init_relations = kwargs.get('init_relations')
        if init_relations is not None:
            self.R = nn.Parameter(init_relations)
        else:
            self.R = nn.Parameter(torch.FloatTensor(relation_dim, embed_dim).uniform_(-.1,.1))
        self.R.my_name = 'R'
        self.R.grad = torch.zeros_like(self.R)
        
        pretrained_ent = kwargs.get('pretrained_entities')
        if pretrained_ent is not None:
            self.E = nn.Parameter(pretrained_ent)
        else:
            self.E = nn.Parameter(torch.FloatTensor(entitiy_dim, embed_dim).uniform_(-.1,.1))
        self.E.my_name = 'E'
        self.E.grad = torch.zeros_like(self.E)
        
        self.loss_function = nn.SoftMarginLoss(reduction='sum')
        self.optim = Adagrad(list(self.parameters()), lr=self.learning_rate)
        
    def forward(self, xs, ys, minibatch_size):
        for loss, grads in self._optim(list(zip(xs, ys)), minibatch_size):
            yield loss, grads
        
    def _optim(self, xys, minibatch_size):
        for self._epoch in range(1, self._max_epochs+1):
            self.loss = 0
            self.optim.zero_grad()
            self.train()
            
            # shuffle training examples
            indices = list(range(len(xys)))
            shuffle(indices)
            
            # store epoch for callback
            self.epoch_start = timeit.default_timer()
            
            # process mini-batches
            lower_iter, upper_iter = count(0, minibatch_size), count(minibatch_size, minibatch_size) 
            for lower, upper in zip(lower_iter, upper_iter):
                # select indices for current batch
                if lower >= len(indices):
                    break

                batch_examples = [xys[idx] for idx in indices[lower:upper]]
                triples,ys = zip(*batch_examples)
                ss,ps,os = zip(*triples)
                ss,ps,os,ys=torch.LongTensor(ss), torch.LongTensor(ps), torch.LongTensor(os), torch.FloatTensor(ys)
                        
                yscores = self._scores(ss, ps, os) # see Holographic Embeddings, eq. 2
                self.loss = self.loss_function(yscores, ys)
                print('loss', self.loss)

                fs = -(ys * torch.sigmoid(-yscores)).unsqueeze(1)
                entity_grad, entity_idxs = self._fn_Entity_Grad(yscores, ss, os, ps, fs)
                relation_grad, relation_idxs = self._fn_Relation_Grad(yscores, ss, os, ps, fs)
                #print('grad rel', relation_grads.shape, torch.sum(relation_grads))
                
                for param in self.parameters():
                    if param.my_name == 'R':
                        self.R.grad = relation_grad
                    
                    if param.my_name == 'E':
                        for col,row_grads in zip(entity_idxs, entity_grad): # FIXME use index_put_
                            self.E.grad[col] = row_grads

                self.optim.step()
                
                #batch_loss, batch_grads = self._process_batch(bxys)
                #yield batch_loss, batch_grads


    def _fn_Entity_Grad(self, yscores, ss, os, ps, fs):
        sparse_indices, Sm, n = grad_sum_matrix(torch.cat((ss, os)))
        combined = torch.cat((fs * ccorr(self.R[ps], self.E[os]),
                              fs * cconv(self.E[ss], self.R[ps])),
                             dim=0)
        grads = torch.mm(Sm, combined) / n.unsqueeze(1)
        return grads, sparse_indices

    def _fn_Relation_Grad(self, yscores, ss, os, ps, fs):
        sparse_indices, Sm, n = grad_sum_matrix(ps)
        grads = torch.mm(Sm, fs * ccorr(self.E[ss], self.E[os])) / n
        return grads, sparse_indices
        
    def _scores(self, ss, ps, os):
        return torch.sum(self.R[ps] * ccorr(self.E[ss], self.E[os]), dim=1)

    def _update(self, g, idx=None):
        self.p2[idx] += g * g
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        self.param[idx] -= self.learning_rate * g / H

def _update_using_Adagrad(self, vals, grad):
    vals += grad * grad
    H = torch.max(torch.sqrt(vals), 1e-7)
    self.param[idx] -=learning_rate * g / H

def normless1(M, idx=None):
    nrm = torch.sum(M[idx] ** 2, axis=1).unsqueeze(1)
    nrm[nrm < 1] = 1
    M[idx] = M[idx] / nrm
    return M
