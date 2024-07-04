import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm

from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence


import math 

from utils import read_config, read_corpus

#debug library
import pdb
import sys
import os

#this file contains the code that is worked and achieved 76%


#multiHeadSelf Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        

        #split embedding into self.head pieces
        values = values.reshape(N,value_len,self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N,query_len,self.heads,self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        #queries shape: (N, query_len, heads, heads_dim)
        #keys shape: (N,key_len, heads, heads_dim)
        #energy shape: (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
            
        
        attention = torch.softmax(energy/ (self.embed_size**(1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N, query_len, self.heads*self.head_dim)
        #attention shape; (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions
        
        out = self.fc_out(out)
        
        return out
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention=SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value,key,query,mask)
        
        x = self.dropout(self.norm1(attention+query))
        
        forward = self.feed_forward(x)
        
        out = self.dropout(self.norm2(forward+x))
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self,embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock,self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x,x,x, tgt_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value,key,query, src_mask)
        return out
                
                
class Decoder(nn.Module):
    def __init__(
      self,
      config,
      ):
        super(Decoder,self).__init__()
        self.config = config
        self.device = config.device
        self.word_embedding = nn.Embedding(config.tgt_vocab_size,config.embedding_size)
        self.position_embedding =  nn.Embedding(config.tgt_max_length,config.embedding_size) #positional_encoding(max_length,embed_size)
        self.layers = nn.ModuleList( 
            [DecoderBlock(config.embedding_size,config.num_heads, config.forward_expansion,config.dropout, config.device)
             for _ in range(config.num_decoder_layers)]
        )
        
        self.fc_out = nn.Linear(config.embedding_size,config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)
      
    def make_tgt_mask(self,tgt):
        N, tgt_len = tgt.shape
        
        if self.config.pre_train_mask:
            tgt_mask = torch.ones((tgt_len,tgt_len)).expand(N,1,tgt_len,tgt_len)
        else:
            tgt_mask = torch.tril(torch.ones((tgt_len,tgt_len))).expand(N,1,tgt_len,tgt_len)
                
        return tgt_mask.to(self.device)
    
    def forward(self, x, enc_out, src_mask,tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x = layer(x, enc_out,enc_out,src_mask,tgt_mask)
        
        out = self.fc_out(x)
        #pdb.set_trace()
        #out = F.softmax(out,dim=2) #compute probabilities
       
        return out
     
    def generate(self, enc_src, src_mask):
        self.eval()  # Set the decoder to evaluation mode
        with torch.no_grad():
            # Initialize the target sequence with the start token
            tgt = torch.full((enc_src.size(0),1),self.config.tgt_bos).to(enc_src.device)
            #tgt = torch.tensor([[self.config.tgt_bos]*enc_src.size(0)], device=enc_src.device)
            #tgt = torch.ones(enc_src.size(0)).long().fill_(self.config.tgt_bos)
            for _ in range(self.config.tgt_max_length):
                tgt_mask = self.make_tgt_mask(tgt)
                logits = self.forward(tgt, enc_src, src_mask, tgt_mask)
                logits = F.log_softmax(logits,dim=2)
                _, predicted = torch.max(logits[:, -1, :], dim=1)

                #constrains

                #if the previous prediction was 2 make the current one aslo 2
                mask = (tgt[:,-1] == 2)
                predicted[mask] = 2
 
      
                # Append the predicted token to the target sequence
                #tgt = torch.cat([tgt, predicted.unsqueeze(0)], dim=1)
                tgt = torch.cat([tgt, predicted.unsqueeze(1)],dim=1)

                # Check if the end token is generated for all batches
                if (predicted == self.config.tgt_eos).all():
                    break
                
            return tgt[:, 1:]  # Remove the start token from the generated sequences


        
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, label_smoothing=0.0,tgt_pad_idx=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.tgt_pad_idx = tgt_pad_idx
        self.label_smoothing = label_smoothing


    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,ignore_index=self.tgt_pad_idx,label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

 
class RADAr(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(RADAr, self).__init__()
        self.config = config 
        
        self.encoder = AutoModel.from_pretrained(config.encoder_model)#output_hidden_states=True)

        self.decoder = Decoder(config)
        
        # Instantiate the FocalLoss with the specified gamma and class_weights
        self.criterion = FocalLoss(gamma=self.config.gamma,label_smoothing=config.label_smoothing,tgt_pad_idx = config.tgt_pad_idx).to(config.device)

        
    def make_src_mask(self,src):
        src_mask = (src != self.config.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #(N,1,1,src_len)
        return src_mask.to(self.config.device)
    
    def make_tgt_mask(self,tgt):
        N, tgt_len = tgt.shape
        
        tgt_mask = torch.tril(torch.ones((tgt_len,tgt_len))
                             ).expand(N,1,tgt_len,tgt_len)
                
        return tgt_mask.to(self.config.device)
    

    
    def forward(self, src, tgt=None,hiera=None):
        src_mask = self.make_src_mask(src['attention_mask'])
        tgt_mask = self.make_tgt_mask(tgt[:,:-1]) if tgt is not None else None

        #tgt_mask = self.make_tgt_mask(tgt) if tgt is not None else None
        enc_out = self.encoder(**src)
        if tgt is not None:
            out = self.decoder(tgt[:,:-1], enc_out.last_hidden_state, src_mask, tgt_mask)
            #out = self.decoder(tgt, enc_out.last_hidden_state, src_mask, tgt_mask)
            #compute loss
            output = out.reshape(-1,out.shape[2])
            #output = F.log_softmax(output,dim=-1)


            tgt = tgt[:,1:].reshape(-1)
            
            loss = self.criterion(output,tgt)
            return loss, out
        else:
            #greedy search
            out = self.decoder.generate(enc_out.last_hidden_state, src_mask)  
            return out
