import json

import pdb

def get_data_as_list(in_file):
    data = []
    with open(in_file, 'r') as sample_file:
        data = sample_file.readlines()
    return data


def create_RADAr_samples(in_file,split="train"):  
    data = get_data_as_list(in_file)
    context_file = open(f"{folder}/{split}.src","w")
    target_file = open(f"{folder}/{split}.tgt","w")

    for line in data:
        dic = eval(line)# convert line into dictionary
        
        # strip, lower case and truncate text to 512 words
        context = dic['token'].strip().lower()[:512]
        context_file.write(context+"\n")
        
        #strip and lower case labels
        labels = ' '.join([element.strip().lower() for element in dic['label']])
        target_file.write(labels+"\n")
        
                    
    context_file.close()
    target_file.close()
                    
                
def crate_sorted_topic_list():
    tmp = [f"{folder}_train_all.json",f"{folder}_test_all.json",f"{folder}_val_all.json"]
        
    all_data = []
    for item in tmp:
        _list = get_data_as_list(item)
        all_data.extend(_list)
    
    topics = {}
    #k,v are the lable and its count in the data
    for sample in all_data:
        sample = eval(sample) #convert str to dic
        for label in sample['label']:
            label = label.strip().lower()
            if label not in topics:
                topics[label]=1
            else:
                topics[label]+=1
                
    #sort dictionary based on its values which is sorting the topics based on frequency
    sorted_topics = dict(sorted(topics.items(), key=lambda x: x[1],reverse=True))

    #create dictionary where key is topic and value is its order
    topic_order = {}
    count = 0
    for k,v in sorted_topics.items():
        topic_order[k] = count
        count +=1
        
    with open(f"{folder}/"+"topic_sorted.json","w") as json_file:
        #Write the data to the file
        json.dump(topic_order, json_file)

    
    
    
folder= "my_model_data" #wos or rcv1-v2

create_RADAr_samples("rcv1_train_all_generated_tl.json",split="train")
create_RADAr_samples("rcv1_test_all_generated.json",split="test")
create_RADAr_samples("rcv1_val_all_generated.json",split="valid")
               
    
    
#create sorted topic file
crate_sorted_topic_list()
    
  
