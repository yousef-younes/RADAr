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
            sent = sent[::-1]
            labels.append(sent)

    return labels

#load label hiera
def load_hiera(hiera_file):
    pseudo_hiera = {}
    with open(hiera_file,"r") as f:
        pseudo_hiera = json.load(f)
    return pseudo_hiera

#this function takes the hierarchy dictionary and return a dicitony whose keys#are childen and value is the direct parent
def get_child_direct_parent_dict(hiera_dict):
    child_parent_dict = {}
    for k,v in hiera_dict.items():
        for child in v:
            child_parent_dict[child]=k
    return child_parent_dict

#take a list of labels ordered from children to parent. It return one path and the remaining unsed labels
def extract_path(sample):
   
    path =[]

    #handle the case when only one root label is present in the ground truth
    if len(sample) == 1:
        path.append(sample[0])
        path.append("<unk>")
        return path, sample
   
    for lbl in sample:
        if len(path) == 0:
            path.append(lbl)
        elif lbl == child_parent_dict[path[len(path)-1]]:
             #if current label is paretn of last label in path, add it
             path.append(lbl)

             if child_parent_dict[lbl] == "Root":
                 path.append("<unk>")
                 break
    #handle the case when a label is left alone and all its parents are removed
    if len(path) == 1:
       while(path[len(path)-1]!="<unk>"):
           path.append(child_parent_dict[path[len(path)-1]])
           if path[len(path)-1]=="Root":
               path[len(path)-1] = "<unk>"
    #remove used labels    
    for lbl in path[:-1]:# last token '<unk>' is not a lable it just signals the end of the path
        if lbl in label_hiera["Root"]:#do not remove root nodes
            continue
        if lbl in sample:
            sample.remove(lbl)
    return path, sample       

#this function takes a list of lables and return true if all of its labels are root level labels
def are_all_paths_extracted(labels):
    for lbl in labels:
        if lbl not in label_hiera["Root"]:
            return False
    return True
         
dataset_dir = "rcv1/" #"wos/" #"rcv1/" #"nyt/" 
split = "valid.tgt"


ground_truth_labels = get_labels(dataset_dir+split)
print(len(ground_truth_labels))
label_hiera = load_hiera(dataset_dir+"pseudo_hiera.json")

print(len(label_hiera))
child_parent_dict = get_child_direct_parent_dict(label_hiera)
print(len(child_parent_dict))


organized_samples = []
for sample in ground_truth_labels:
    sample_paths = []
    
    #handle the case when only one root label is present in the ground truth
    if len(sample) == 1:
        sample_paths.append(sample[0])
        sample_paths.append("<unk>")
    while not are_all_paths_extracted(sample):
        #print(sample) 
        prev_num_labels = len(sample)
        #path, remaining_lables = extract_path(remaining_labels)
        path, sample = extract_path(sample)
        #add extracted path to sample list of paths if it is not a single label which happens when some labels are removed from the current label list
        if len(sample) < prev_num_labels:
            if len(path) <2:
                pdb.set_trace()

            sample_paths.extend(path)
    #handle the case when all labels are root ones
    if len(sample_paths) == 0 and len(sample)>0:
       for lbl in sample:
           sample_paths.extend([lbl,"<unk>"])
    #print("********************************")
    #if len(sample_paths) ==0:
        #pdb.set_trace()
    organized_samples.append(sample_paths)


def save_file(file_name, organized_labels):
    with open(file_name,"w") as f:
        for sample in organized_labels:
            f.write(" ".join([l for l in sample])) 
            f.write("\n")
save_file(dataset_dir+"organized/"+split,organized_samples)


#compute the longest label list that should be predicted for a sample 
#for nyt it is 99
max_lbl = 0
with open(dataset_dir+"organized/"+split,"r") as f:
    lines = f.readlines()
    for line in lines:
        lbls = line.strip().split(" ")
        if len(lbls) > max_lbl:
           max_lbl = len(lbls)
    print(f"Max label length is {max_lbl}\n")

