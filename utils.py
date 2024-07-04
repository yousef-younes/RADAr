import yaml
import os
import time
import sys
import json 
import math
from typing import List
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import pdb



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'),Loader=yaml.FullLoader))


def get_logger(file):
    make_sure_dir_exits_or_create(file)
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log

# takes a file name and make sure to create its directory if it does not exist
def make_sure_dir_exits_or_create(file_path):
     # Extract directory path
    directory = os.path.dirname(file_path)

    # Check if directory exists
    if not os.path.exists(directory):
        # Create missing directories recursively
        os.makedirs(directory)


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        labels = [e[1] for e in examples]

        yield src_sents, labels


def read_corpus(file_path,text_type=None):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    
    #i = 0
    for line in open(file_path):
        if text_type == "tgt":
            sent = line.strip().split(' ')
            sent = sent[::-1] # reverse the list
            sent = sent[1:] # remvoe the first token
            sent.append("<unk>") # add <unk> token to the end of the list

        else:   
            sent = line.strip()
            #use tfidf to select the best 512 tokens
            #sent = feature_selection(sent) 
        data.append(sent)
    
    return data

#this function takes the split name and return its data in the form [(context,labels),...]
def get_data(split,config):
    if split=="train":
        train_data_src = read_corpus(config.train_src)#[:1000]
        train_data_tgt = read_corpus(config.train_tgt,text_type="tgt")#[:1000]
        return pair_lists(train_data_src,train_data_tgt)
    elif split == "val":
        val_data_src = read_corpus(config.val_src)#[:100]
        val_data_tgt = read_corpus(config.val_tgt,text_type="tgt")#[:100]
        return pair_lists(val_data_src,val_data_tgt)
    elif split=="test":
        test_data_src = read_corpus(config.test_src)#[:100]
        test_data_tgt = read_corpus(config.test_tgt,text_type="tgt")#[:100]
        return pair_lists(test_data_src,test_data_tgt)
    else:
        print(f"{split} is not valid split name.")

        
def pair_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")

    paired_list = []
    for i in range(len(list1)):
        paired_list.append((list1[i], list2[i]))

    return paired_list

# save predictions along with original labels in a text file
def save_pred_org_lbls(file_name,predicted_lbls,original_lbls):
    make_sure_dir_exits_or_create(file_name)
    
    with open(file_name, "w") as f:
        f.write("sampels are separated by double lines, the first line corresponds to the predicted lables while the second line corresponds to the original labels\n")
        f.write("****************************************************\n\n")
        for p, o in zip(predicted_lbls, original_lbls):
            f.write(f"{p}\n{o}\n\n")

#this function receives a list of lists representing the model geneated batch output and produces a list of lists of the same size but with 
#the pad, bos, eos and unk tokens removed
def clean_output_labels(batch):
    cleaned_labels = []
    for sample in batch:
        tmp = []
        for token in sample:
            if token in [0,1,2,3]:# this is the list of undesirable tokens
                continue
            tmp.append(token)
        cleaned_labels.append(tmp)
    return cleaned_labels

#this function receives a batch of labels i.e., list of lists and flatten the labels into one list
def flaten_labels(labels):
    temp = []
    for lbls in labels:
        temp.extend(lbls)
    return temp

#this function loads the label hierarchy indexes into a dictionary
def load_json_file(hiera_file):
    #load hierarchy
    label_hiera={}
    with open(hiera_file,"r") as hiera_file:
        label_hiera = json.load(hiera_file)
    return label_hiera

#laod_pickle_fiel
def load_pickle_file(file_path):
    data = {}
    with open(file_path,"rb") as file:
        data = pickle.load(file)
    return data

#this function receives a batch of labels i.e., list of lists and returns the labels read for metric computation i.e., in the form [[0,.....],[0,1,0,0,...]]
def prepare_labels_for_matrics_computation(batch,num_labels):
    #num_labels =141
    batch_encoding = np.zeros((len(batch),num_labels), dtype=int)
    for i,sample in enumerate(batch):
        for idx in sample:
            if idx not in [0,1,2,3]:#skip pad,bos,eos and unk tokens
                batch_encoding[i,idx-4]=1
                
    return batch_encoding.tolist()
        


# this function takes two lists of labels and compute the matrics on them
def compute_metrics(origin, preds,log_printer):
    # calculate F1-score
    f1_score_val = f1_score(origin, preds, average=None)
    # calculate macro-average f1-score
    macro_f1_score = f1_score(origin, preds, average='macro')

    micro_f1_score = f1_score(origin, preds, average='micro')
    log_printer(f"f1 score:{f1_score_val}\nmacro f1-score:{macro_f1_score}\nmicro_f1_score:{micro_f1_score}\n\n")
    
    
#this function set the random seeds so that we can get the same resutls when running the same experiment in the future
def set_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    #set python built-in pseudo random generator at a fixed value
    random.seed(seed_value)
    #set numpy pseudo random generator at a fixed value
    np.random.seed(seed_value)
    #when using tensor flow 
    #tf.set_random_seed(seed_value)



#initialize the embedding dictionary of the decoder
def initialize_embeddings(config,decoder_input_embedding):
    #laod decoder vocab file
    decoder_vocab = load_json_file(config.label_file)

    #laod text label - pseudo label dicitonay
    text_pseudo_dict = load_pickle_file(config.text_pseudo_file)
    
    pseudo_text_dict = {}
    #creae pseudo_text_dec
    for k,v in text_pseudo_dict.items():
        pseudo_text_dict[v.strip().lower()] = k
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
    model = AutoModel.from_pretrained(config.encoder_model)
    word_embeddings = model.get_input_embeddings() #.roberta.embeddings.word_embeddings
    
    for lbl_pseudo, lbl_index_in_decoder in decoder_vocab.items():
        #handle special tokens <s> </s> <pad> <unk>
        if lbl_index_in_decoder < 4:
            text_to_encode = lbl_pseudo
        else:
            text_to_encode = pseudo_text_dict[lbl_pseudo]
        
        token_ids = torch.tensor(tokenizer.encode(text_to_encode))
        emb = word_embeddings(token_ids)
        emb_mean = emb.mean(dim=0) #[emb_size]
        decoder_input_embedding.weight.data[lbl_index_in_decoder]= emb_mean

    return decoder_input_embedding


def feature_selection(text):
    
    # Load RoBERTa tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize and encode the entire long text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)

    # Extract features using a sliding window
    feature_importance = []

    # Calculate TF-IDF scores for the current window
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf_matrix = tfidf_vectorizer.fit_transform([tokenized_text])

    # Assuming you want to use TF-IDF scores as feature importance
    feature_importance = np.array(tfidf_matrix.sum(axis=0)).squeeze()

    N= 510 if len(feature_importance)>510 else len(feature_importance)
    # Select the top N features based on TF-IDF importance
    top_features_indices = np.argsort(feature_importance)[-N:][::-1]

    # Select top N tokens based on TF-IDF importance
    selected_tokens = [tokenizer.decode(tokenized_text[i]) for i in top_features_indices]

    # Combine selected tokens into a new input for RoBERTa
    selected_text = " ".join(selected_tokens)
    return selected_text

def old_feature_selection(text):
    
    # Load RoBERTa tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize and encode the entire long text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)

    # Define the window size
    window_size = 510  # Adjust the window size based on your requirements

    # Extract features using a sliding window
    feature_importance = []
    for i in range(0, len(tokenized_text), window_size):
        window = tokenized_text[i:i+window_size]

        # Calculate TF-IDF scores for the current window
        tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        tfidf_matrix = tfidf_vectorizer.fit_transform([window])

        # Assuming you want to use TF-IDF scores as feature importance
        window_feature_importance = np.array(tfidf_matrix.sum(axis=0)).squeeze()
        #print(window_feature_importance)
        #print(window_feature_importance.shape)
        if window_feature_importance.shape == ():
            continue
            #feature_importance.append(window_feature_importance)
        feature_importance.extend(window_feature_importance)

    N= 510 if len(feature_importance)>510 else len(feature_importance)
    # Select the top N features based on TF-IDF importance
    top_features_indices = np.argsort(feature_importance)[-N:][::-1]

    # Select top N tokens based on TF-IDF importance
    selected_tokens = [tokenizer.decode(tokenized_text[i]) for i in top_features_indices]

    # Combine selected tokens into a new input for RoBERTa
    selected_text = " ".join(selected_tokens)
    pdb.set_trace()
    return selected_text
