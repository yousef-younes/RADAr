import json
import pdb
import sys

#load target file
def get_labels(file):
    labels = []

    with open(file,"r") as f:
        lines =f.readlines()
        for line in lines:
            sent = line.strip().split(' ')
            labels.append(sent)

    return labels

#load label hiera
def load_hiera(hiera_file):
    pseudo_hiera = {}
    with open(hiera_file,"r") as f:
        pseudo_hiera = json.load(f)
    return pseudo_hiera

#this function takes two lists. the first list is the labels and the second list is the labels in aspecific level in the hierarchy and return their matches
def get_level_labels(sample_lbls,labels):
    level_labels = []
    for lbl in sample_lbls:
        if lbl in labels:
            level_labels.append(lbl)
    return level_labels  
    
def get_next_level_labels(lbls):
    hiera_level_lbls = []
    for lb in lbls:
        if lb in label_hiera.keys():
            tmp_lbls = label_hiera[lb]
            hiera_level_lbls.extend(tmp_lbls)
    return hiera_level_lbls
        
def remove_used_items(original_list, used_items):
    for item in used_items:
        if item not in original_list:
            pdb.set_trace()
        original_list.remove(item)
    return original_list
           
#this function takes a list of labels and returns it organized level wise i.e., the root nodes at the begining then the labels from the second level and so on
def organize_labels(labels):
    organized_labels = []
    cur_level_labels = label_hiera["Root"]
    while True:
        level_labels = get_level_labels(labels,cur_level_labels)
        
        #remove organized labels
        labels = remove_used_items(labels,level_labels)
        #add tokens to seperate levels in the hierarchy
        if len(level_labels) > 0:
            level_labels.append("<unk>")
        organized_labels.extend(level_labels)
        cur_level_labels = get_next_level_labels(cur_level_labels)
        if len(labels) == 0:
            break
    #assert len(organized_labels) == len(labels)
    return organized_labels
        

#this function takes the hierarchy dictionary and return a dicitony whose keys#are childen and value is the direct parent
def get_child_direct_parent_dict(hiera_dict):
    child_parent_dict = {}
    for k,v in hiera_dict.items():
        for child in v:
            child_parent_dict[child]=k
    return child_parent_dict

#this function takes a list of lables and return true if all of its labels are root level labels
def are_all_paths_extracted(labels):
    for lbl in labels:
        if lbl not in label_hiera["Root"]:
            return False
    return True
         
dataset_dir = "wos/" #"wos/" #"rcv1/" #"nyt/" 
split = "val.tgt"


ground_truth_labels = get_labels(dataset_dir+split)
print(len(ground_truth_labels))
label_hiera = load_hiera(dataset_dir+"pseudo_hiera.json")

print(len(label_hiera))
child_parent_dict = get_child_direct_parent_dict(label_hiera)
print(len(child_parent_dict))


organized_samples = []
for sample in ground_truth_labels:
    organized_labels = organize_labels(sample)
    organized_samples.append(organized_labels)


def save_file(file_name, organized_labels):
    with open(file_name,"w") as f:
        for sample in organized_labels:
            f.write(" ".join([l for l in sample])) 
            f.write("\n")
save_file(dataset_dir+"organized_level_wise/"+split,organized_samples)


#compute the longest label list that should be predicted for a sample 
#for nyt it is 99
max_lbl = 0
with open(dataset_dir+"organized_level_wise/"+split,"r") as f:
    lines = f.readlines()
    for line in lines:
        lbls = line.strip().split(" ")
        if len(lbls) > max_lbl:
           max_lbl = len(lbls)
    print(f"Max label length is {max_lbl}\n")

