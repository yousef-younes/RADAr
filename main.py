import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer , get_linear_schedule_with_warmup
import math

from utils import read_config, get_data, batch_iter, save_pred_org_lbls, clean_output_labels, compute_metrics, flaten_labels, prepare_labels_for_matrics_computation, get_logger,set_random_seeds,make_sure_dir_exits_or_create

from model import RADAr
from label_tokenizer import LabelTokenizer


#debug library
import pdb
import sys
import os
import multiprocessing as mp 

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

def train(config,model,log_printer): 
    #load train and validate data
    train_data = get_data(split="train",config=config)
    val_data = get_data(split="val",config=config)

    # Set the number of gradients to accumulate before optimization step (NEW)
    accumulation_steps = config.accumulation_steps 
    
    
    num_training_steps_per_epoch = math.ceil(len(train_data)/config.batch_size)
    
    
    num_training_steps = num_training_steps_per_epoch // accumulation_steps
    num_total_training_steps = config.num_epochs * num_training_steps_per_epoch


    
    optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': float(config.enc_lr)},
    {'params': model.decoder.parameters(), 'lr': float(config.dec_lr)}
    ])
    

        
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(config.lr_patience), verbose=True)

    saved_epoch = 0

    if config.continue_training:
        #load stored dictionary
        checkpoint = torch.load(config.trained_model_file)
        #load model
        model.load_state_dict(checkpoint['model']) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        
    #freeze encoder
    #for param in model.encoder.parameters():
        #param.requires_grad = False
       
    #for name, param in model.named_parameters():
        #print(name, param.requires_grad)

    # Count the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    log_printer(f"Total number of parameters in the model: {total_params}\n")
    
    # Compute the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_printer(f"Total Trainable Parameters:{total_params}\n\n")
    
        
    pad_idx = config.src_pad_idx

    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
    # Define the user-defined vocabulary size and other parameters
    decoder_tokenizer = LabelTokenizer(config.label_file,config.hiera_file,max_length = config.tgt_max_length)


    #the number of trianing steps 
    global_step = 0
    current_epoch_steps = 0
    
 
    total_loss = float('inf') #0.030390 
    patience = float(config.patience)
    once_switch= True    
    epoch = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
                
        train_progress_bar = tqdm(batch_iter(train_data, config.batch_size,config.shuffle), total=num_training_steps_per_epoch-1, desc="Training")
        for step, (x,y) in enumerate(train_progress_bar):
            

            contexts = encoder_tokenizer(x,padding=True, truncation=True,max_length=config.src_max_length,return_tensors='pt') 

            
            #tokenize target sequences and get target token IDs
            labels = decoder_tokenizer.encode_labels(y,config.tgt_max_length,padding=True, return_tensors='pt',preprocess=False)      
 
            contexts = contexts.to(config.device)
            labels = labels.to(config.device)
            

            loss, output = model(contexts,labels)
           
                        
            #writer.close()        
            loss = loss / accumulation_steps  # Scale the loss to account for gradient accumulation (NEW)

            train_loss += loss

            loss.backward()
             
            if (step + 1) % accumulation_steps == 0:
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                #model.zero_grad()

            
            current_epoch_steps += 1
            global_step +=1
            
        avg_valid_loss,valid_loss = validate(config,val_data,model,encoder_tokenizer,decoder_tokenizer)
		
	#compute the loss for every half epoch
        avg_train_loss = train_loss / len(train_data)           


        log_printer("\nEpoch:{:n} \tStep: {:n}/{:n} \tEncoder LR:{:.6f} \tDecoder LR:{:.6f} \tAvg Training Loss: {:.6f} \tAvg Valid Loss: {:.6f}\n".format(epoch,global_step,num_total_training_steps,optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],avg_train_loss,avg_valid_loss))
		
	#change learning rate based on validation loss
        scheduler.step(avg_valid_loss)

        if total_loss > avg_valid_loss:
            log_printer("Resutls improve and model is saved\n************************************\n")
            total_loss = avg_valid_loss
            dict_to_save = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
             }
            #this variable maintain the epoch in which the model was saved
            saved_epoch= epoch 

            #create models folder if not available
            make_sure_dir_exits_or_create(config.model_file+str(saved_epoch))
            
            torch.save(dict_to_save, config.model_file+str(saved_epoch))
            #torch.save(model.state_dict(), config.model_file+str(saved_epoch))
            patience=float(config.patience)
        else:
            patience = patience-1
            if patience ==0:
                log_printer(f"\nThe model did not improve for {config.patience} epochs. So we end the training process")
                return saved_epoch 
            #zero the encoder learning rate to focus on the decoder
            if  optimizer.param_groups[0]['lr'] < 0.0000005 and once_switch:
                once_switch = False
                #load best stored dictionary
                checkpoint = torch.load(config.model_file+str(saved_epoch))
                #load model
                model.load_state_dict(checkpoint['model']) 
                once_switch = False
                log_printer("***********From Here the encoder is frozen and the decoder continue learning alone*************\n")
                for param in model.encoder.parameters():
                    param.requires_grad = False
     

        #clear losses
        train_loss = 0
        valid_loss = 0
        avg_valid_loss = 0
        avg_train_loss= 0


    return saved_epoch
                    
                    
                
            

def validate(config,val_data,model,encoder_tokenizer,decoder_tokenizer):
    valid_loss = 0
    model.eval()
    with torch.no_grad():
            
        num_validation_steps = math.ceil(len(val_data)/config.batch_size)
        val_progress_bar = tqdm(batch_iter(val_data, config.batch_size), total=num_validation_steps-1, desc="Validating")
        
        for x,y in val_progress_bar:
            contexts = encoder_tokenizer(x,padding=True, truncation=True,return_tensors='pt')            

            #tokenize target sequences and get target token IDs
            labels = decoder_tokenizer.encode_labels(y,config.tgt_max_length,padding=True, return_tensors='pt',preprocess=False)  

            contexts = contexts.to(config.device)
            labels = labels.to(config.device)

            #forward propogation
            loss, output = model(contexts,labels)#[:,:-1])

            valid_loss += loss
            
    return valid_loss/len(val_data), valid_loss
               

def test(config,model,log_printer,saved_epoch):
    #load data
    test_data = get_data(split="test",config=config)
    num_test_steps = math.ceil(len(test_data)/config.batch_size)
    test_progress_bar = tqdm(batch_iter(test_data, config.batch_size), total=num_test_steps, desc="Tesing")
    #load hierarchy
    device = torch.device(f'cuda:{config.device}')  # Specify the desired GPU (GPU 1 in this case)
    checkpoint = torch.load(config.model_file+str(saved_epoch))
    # Load the state_dict into the model
    model.load_state_dict(checkpoint["model"]) 
    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
    # Define the user-defined vocabulary size and other parameters
    decoder_tokenizer = LabelTokenizer(config.label_file,config.hiera_file,max_length = config.tgt_max_length)

    model.to(config.device)
    model.eval()
    
    #these lists are used to print predicted and original tokens
    predicted_lbls, original_lbls = [], []
    #these lists are used to compute the metricis
    metric_original_labels, metric_prediction_labels = [], []
    for x,y in test_progress_bar:
    #for x, y in tqdm(batch_iter(test_data, config.batch_size)):
        contexts = encoder_tokenizer(x,padding=True, truncation=True,return_tensors='pt')            
                             
        contexts = contexts.to(config.device)
        #labels = labels.to(config.device)
         
        #generate labels
        with torch.no_grad():
            outputs = model(src=contexts,tgt=None) #,hiera=label_hiera)
        #move outputs to cpu and convert them to list of lists
        outputs = outputs.to('cpu').tolist()
        #decode predicted labels
        batch_predicted_labels = decoder_tokenizer.decode_labels(outputs, skip_special_tokens=True,postprocess=False)
        #decode original labesl 
        #batch_original_labels = decoder_tokenizer.decode_labels(labels, skip_special_tokens=True)
        
        predicted_lbls.extend(batch_predicted_labels)
        original_lbls.extend(y)
        
        #prepare labels to compute metrics
        #There is problem here try to fix it the labels should be of the form [[1,0,1],[1,1,0],....[0,0,1]]
        num_labels= config.tgt_vocab_size -4 #4 accounts for <s> </s> <pad> <unk> tokens
        #tokenize original target sequences and get target token IDs but without padding,eos and sos
        org_labels = decoder_tokenizer.encode_labels(y,config.tgt_max_length, eval_mode = True,preprocess=False)
        ready_org_labels = prepare_labels_for_matrics_computation(org_labels,num_labels)#4 account for <s> </s> <pad> <unk> tokens
        metric_original_labels.extend(ready_org_labels)
        #clearn predicted labels
        #outputs = decoder_tokenizer.encode_labels(batch_predicted_labels,config.tgt_max_length,eval_mode=True,preprocess=False)
        ready_pred_labels = prepare_labels_for_matrics_computation(outputs,num_labels)
        metric_prediction_labels.extend(ready_pred_labels)

    
    # save predictions along with original labels in a text file
    save_pred_org_lbls(config.result_file,predicted_lbls,original_lbls)
    

    #compute f1 scores
    compute_metrics(metric_original_labels, metric_prediction_labels,log_printer)
    



def main():
    #read the argument passed to the program
    dataset = sys.argv[1]
    #load config file
    config = read_config(f"{dataset}_config.yaml")

    print(config)

    set_random_seeds(config.random_seed)
        
    #create model
    model = RADAr(config = config).to(config.device)
   
     
    log_printer = get_logger(config.log_file)
    
 
    log_printer("\n\n**********************************\n") 
    log_printer(str(config))
    log_printer("\n************************************\n\n")
    #train model
    saved_epoch = train(config,model,log_printer)
    
    #test model
    test(config,model,log_printer,saved_epoch)
    
    
if __name__ == main():
    main()
