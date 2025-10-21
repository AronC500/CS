# by  
# 2024.3.30

"""
This code is modified from Adoni's repository.
https://github.com/Adoni/word2vec_pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

 
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dimension, dropout=0.5):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        
        self.u_embedding = nn.Embedding(self.vocab_size, self.emb_dimension, sparse=True)
        self.v_embedding = nn.Embedding(self.vocab_size, self.emb_dimension, sparse=True)
        self.dropout = nn.Dropout(dropout)
        
        
        initrange = 0.5 / self.emb_dimension
        self.u_embedding.weight.data.uniform_(-initrange, initrange)
        self.v_embedding.weight.data.uniform_(-0,0)
        # self.v_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        pos_embu = self.u_embedding(pos_u)
        pos_embv = self.v_embedding(pos_v)
        pos_score = torch.sum(torch.mul(pos_embu, pos_embv).squeeze(), dim=1)
        pos_score2 = F.logsigmoid(pos_score)

        neg_embv = self.v_embedding(neg_v)
        neg_score = torch.bmm(neg_embv, pos_embu.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1*neg_score)

        batch_loss = -1 * (torch.sum(pos_score2) + torch.sum(neg_score))
        pos_pred = torch.sigmoid(pos_score) > 0.5
        batch_correct = torch.sum(pos_pred).item()
        batch_total = pos_pred.size(0)

        return batch_loss, batch_correct, batch_total