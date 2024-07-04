import torch
import json
import codecs
from utils import load_json_file

import pdb

class LabelTokenizer():
    
    def __init__(self,label_file,hiera_file,max_length):
        self.label_dict = self.create_label_dic(label_file)
        self.index_label = self.create_index_label_dic()
        #load hierarchy file
        self.hiera = load_json_file(hiera_file)
        #self.parent_child_dict = self.create_parent_child_dict()
        self.child_parent_dict = self.create_child_parent_dict() 
        #specify the number of labels
        self.num_labels = len(self.label_dict)
        self.max_length = max_length
  
    #K is the label token and v is its index
    def create_label_dic(self,label_file):
        with codecs.open(label_file, 'r', 'utf-8') as f:
            label_dict = json.load(f)
        return label_dict
    
    #k,v where k is the token index and v is the label token
    def create_index_label_dic(self):
        index_label = {}
        for k,v in self.label_dict.items():
            index_label[v]= k
        return index_label
    
    #reconstruc paths to the root
    def postprocess(self,labels):
        postprocessed_labels = []
        for label in labels:
            postprocessed_labels.append(label)
            postprocessed_labels.extend(self.get_label_parents(label))
            
            #postprocessed_labels.append("<unk>")
            #if label in self.child_parent_dict:
               #parent = self.child_parent_dict[label]
               #postprocessed_labels.append(parent)
               #postprocessed_labels.append("<unk>")
            
        return set(postprocessed_labels) #remove duplicate labels

    #receive a tensor of label indexes and returns a list of corresponding tokens
    def decode_labels(self, samples,skip_special_tokens=False,postprocess=False):
        tmp = []
        #loop through the resutls samples by sample
        for sample_tgt in samples:
            sample_labels = []
            #decode the tokens of one sample
            for token in sample_tgt:
                if skip_special_tokens & (token in [0,1,2]):#,3]):
                    if token in [2]: #if eos is found ignore the remaining tokens
                        break
                    elif token in [0,1]:#,3]:
                        continue
                sample_labels.append(self.index_label[token])
            if postprocess:#add parents up to the root
                sample_labels = self.postprocess(sample_labels)
            tmp.append(sample_labels)
        return tmp

    #create parent child dictionary
    def create_parent_child_dict(self):
        parent_child_dict = {}
        for k,v in self.hiera.items():
            for child in v:
                parent_child_dict[k] = child
        return parent_child_dict

        
    #create child parent dict, k,v are child parent labels respectively
    def create_child_parent_dict(self):
        child_parent_dict = {}
        for k,v in self.hiera.items():
            for child in v:
                if child in child_parent_dict:
                    print(child)
                    print("Error in Hierarchy")
                child_parent_dict[child]=k
        return child_parent_dict

    
    #get parents of a label up to the root 
    def get_label_parents(self,label):
        
        parents = []
        while True: #label != "Root":
            label = self.child_parent_dict[label]
            if label == "Root":
                break
            parents.append(label)
        return parents
   
    #given a label it returns its childen when it has
    def get_direct_children(self,label):
        if label in self.hiera:
            return self.hiera[label]

    #get children of a label
    def get_label_children(self,label):
        possible_children = []
        children = get_direct_children(label)
        if len(children) > 0:
            possible_children.extend(children)
        return children


    #use hierarchy to remove hierarchy-predictable labels
    def preprocess(self,labels):
        leaf_labels = 0
        important_labels=[]
        covered_labels = []
        #add a label when there is no direct child of it in the list
        for label in labels:  
            if label == '<unk>':
                continue
            if label in covered_labels:
                continue 
            important_labels.append(label)
            covered_labels.extend(self.get_label_parents(label))
        return important_labels
                
    #receives a batch of labels and returns its encoding
    def encode_labels(self,sample,max_length,padding=False,return_tensors=None,eval_mode = False,preprocess=False):
        tmp = []
        for labels in sample:
            if preprocess:
                labels = self.preprocess(labels)
            label_codes=[]

            #add begin of sequence token if the encoding is not for evaluation
            if not eval_mode:
                label_codes = [self.label_dict['<s>']] #add start token

            for l in labels:
                if l not in self.label_dict.keys():
                    pdb.set_trace()
                l = l.strip().lower()
                label_codes.append(self.label_dict[l])

            #add end of sequence token if the encoding is not for evaluation
            if not eval_mode:
                label_codes.append(self.label_dict['</s>']) #add end token

            #if the sequence is short pad to max_length
            if padding & (len(label_codes) < self.max_length) :
                label_codes += [self.label_dict['<pad>']]*((self.max_length)-len(label_codes))
                #label_codes.extend([pad_token]*((self.num_labels+2)-len(label_codes)))  
            tmp.append(label_codes)
            
        if return_tensors=="pt":
            return torch.tensor(tmp)
        else:
            return tmp
        
    def get_label_embedding_size(self):
        return self.num_labels  #tokens are labels,start token, end token and pad token
    
  

#lt = LabelTokenizer("../SGM/data/topic_sorted.json")
#print(lt.get_label_embedding_size())
#print(lt.encode_labels(["medical","lymphoma"]))
      
 


            
        
        
        
