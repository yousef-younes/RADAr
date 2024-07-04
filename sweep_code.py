import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from tqdm import tqdm


from transformers import AutoModel
from transformers import AutoTokenizer , get_linear_schedule_with_warmup
import math

#my code
from utils import read_config, get_data, batch_iter, save_pred_org_lbls, clean_output_labels, compute_metrics, flaten_labels, prepare_labels_for_matrics_computation, get_logger
from label_tokenizer import LabelTokenizer

import torch
import torch.nn

#hyperparameter optimization
import wandb

#debug library
import pdb
import sys
import os

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
      tgt_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      device,
      max_length,
      ):
        super(Decoder,self).__init__()
        self.config = config
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size,embed_size)
        self.position_embedding =  nn.Embedding(max_length,embed_size) #positional_encoding(max_length,embed_size)
        self.layers = nn.ModuleList( 
            [DecoderBlock(embed_size,heads, forward_expansion,dropout, device)
             for _ in range(num_layers)]
        )
        
        self.fc_out = nn.Linear(embed_size,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
      
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
        return torch.mean(focal_loss)

 
        
class RADAr(nn.Module):
    def __init__(
        self,
        config,
        #ls_sweep,
        #gamma,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size,
        num_decoder_layers,
        forward_expansion,
        num_heads,
        dropout,
        device,
        src_max_length,
        tgt_max_length,
    ):
        super(RADAr, self).__init__()
        self.config = config 
        
        self.encoder = AutoModel.from_pretrained(config.encoder_model)#output_hidden_states=True)

        self.decoder = Decoder(config,
                               tgt_vocab_size,
                               embed_size,
                               num_decoder_layers,
                               num_heads,
                               forward_expansion,
                               dropout,
                               device,
                               tgt_max_length
                              )
        #self.src_pad_idx = src_pad_idx
        #self.tgt_pad_idx = tgt_pad_idx
        #self.device = device
        
        
        # Instantiate the FocalLoss with the specified gamma and class_weights
        self.criterion = FocalLoss(gamma=config.gamma, weight=None,label_smoothing=config.label_smoothing,tgt_pad_idx = config.tgt_pad_idx).to(device)
        #self.criterion = nn.CrossEntropyLoss(ignore_index=config.tgt_pad_idx,label_smoothing=config.label_smoothing)

        
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
            #pdb.set_trace() 
            #out = self.decoder(tgt, enc_out.last_hidden_state, src_mask, tgt_mask)
            #compute loss
            output = out.reshape(-1,out.shape[2])
            output = F.log_softmax(output,dim=-1)

            #output  = torch.mean(F.log_softmax(out,dim=-1),dim=1)
            #output = F.softmax(output,dim=-1) #compute probabilities

            tgt = tgt[:,1:].reshape(-1)
            
            loss = self.criterion(output,tgt)
            return loss, out
        else:
            #beam search
            #out = self.decoder.generate_beam(enc_out.last_hidden_state, src_mask,beam_width=self.config.beam_width)
            #greedy search
            out = self.decoder.generate(enc_out.last_hidden_state, src_mask)  # Or any other method to generate output
            #custom search
            #out = self.decoder.hiera_generation(enc_out.hidden_states[11],src_mask,hiera)
            return out



    
def train(_config=None):
    
    config = read_config("nyt_config.yaml")
    with wandb.init(config=_config):
        
        config.label_smoothing = wandb.config.label_smoothing
        config.dropout = wandb.config.dropout
        config.gamma = wandb.config.gamma   
        #create model
        model = RADAr(
        config = config,
        #ls_sweep = wandb.config.label_smoothing,#use wandb parameters
        #src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        src_pad_idx=config.src_pad_idx,
        tgt_pad_idx=config.src_pad_idx,
        embed_size = config.embedding_size,
        #num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        forward_expansion=config.forward_expansion,
        num_heads= config.num_heads,
        dropout =config.dropout, #wandb.config.dropout,
        device = config.device,
        src_max_length = config.src_max_length,
        tgt_max_length= config.tgt_max_length, ).to(config.device)
        
        train_data = get_data(split="val",config=config)
    
        # Set the number of gradients to accumulate before optimization step (NEW)
        accumulation_steps = config.accumulation_steps 
        
        
        num_training_steps_per_epoch = math.ceil(len(train_data)/config.batch_size)
        
        
        num_training_steps = num_training_steps_per_epoch // accumulation_steps
        num_total_training_steps = wandb.config.num_epochs * num_training_steps_per_epoch
        
        
        optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': float(config.enc_lr)},
        {'params': model.decoder.parameters(), 'lr': float(config.dec_lr)}
        ])
        
        
        #freeze encoder
        #for param //in model.encoder.parameters():
            #param.requires_grad = False
            
        #for name, param in model.named_parameters():
            #print(name, param.requires_grad)
    
        # Count the total number of parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        
    
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
            
        pad_idx = config.src_pad_idx
    
        encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
        # Define the user-defined vocabulary size and other parameters
        decoder_tokenizer = LabelTokenizer(config.label_file,config.hiera_file,max_length = config.tgt_max_length)
        
    
        #the number of trianing steps 
        global_step = 0
        current_epoch_steps = 0
        
     
        total_loss =  float('inf') 
        for epoch in range(wandb.config.num_epochs):
            model.train()
            train_loss = 0
                    
            train_progress_bar = tqdm(batch_iter(train_data, config.batch_size,config.shuffle), total=num_training_steps_per_epoch-1, desc="Training")
            for step, (x,y) in enumerate(train_progress_bar):
                
                contexts = encoder_tokenizer(x,padding=True, truncation=True,max_length=config.src_max_length,return_tensors='pt') 
                
                #tokenize target sequences and ge:wt target token IDs
                labels = decoder_tokenizer.encode_labels(y,config.tgt_max_length,padding=True, return_tensors='pt')      
     
                contexts = contexts.to(config.device)
                labels = labels.to(config.device)
                
    
                
                loss, output = model(contexts,labels)
    
                #loss = loss / accumulation_steps  # Scale the loss to account for gradient accumulation (NEW)
    
                train_loss += loss
    
                loss.backward()
                 
                if (step + 1) % accumulation_steps == 0:
                    
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    
                    optimizer.step()
    
                    optimizer.zero_grad()
                
                current_epoch_steps += 1
                global_step +=1

                if step %10 ==0:
                    wandb.log({"batch loss": loss, "step":step})
    
        		
    		
    		
    	    #compute the loss for every half epoch
            avg_valid_loss = train_loss / len(train_data) 
            wandb.log({"loss": avg_valid_loss, "epoch": epoch})   
    
    		
           
            #clear losses
            train_loss = 0 
            avg_valid_loss =0
                    
def get_sweep_dictionary():
    sweep_config = {
        'method': 'grid'
    }
    
    metric={
        'name':'loss',
        'goal': 'minimize'
    }
    
    sweep_config['metric'] = metric
    
    parameters_dict = {
        'enc_lr':{
          'values':[0.0005,0.00005,0.000005] #0.00005,0.000005,0.00005,0.0005,0.005,0.000005]
          #'min': 0.000005,
          #'max': 0.0005
        },
        'dec_lr':{
          'values':[0.0003, 0.00003,0.000003]#.0001, 0.0003, 0.001,0.01]
          #'min': 0.00003, #3e-5
          #'max': 0.003 #3e-3
        },
        'batch_size':{
          'values':[8,16,32,48,64]
        },
        'dropout':{
           'values':[0.2,0.4]
        },
        'label_smoothing':{ 
          'values': [0,0.1,0.5]
         },
       'gamma':{
       'values': [0,1,2,3,4]
       }
    }
    
    sweep_config['parameters'] = parameters_dict
    
    
    parameters_dict.update({
        'num_decoder_layers':{'values': [1,2,3,4]},
        'forward_expansion':{'values': [2,4,8]},
        'num_heads':{'values': [4,8,12]},
        'num_epochs':{'value': 2}
    })

    return sweep_config
    

def main():
    #load config file
    config = get_sweep_dictionary()


    #start the sweep
    sweep_id = wandb.sweep(config, project ="nyt_hype")
    wandb.agent(sweep_id, train)#,count=20)
    
    
    #close wandb 
    wandb.finish()

    
if __name__ == main():
    main()
