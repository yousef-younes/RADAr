import json

import pdb

def get_data_as_list(in_file):
    data = []
    with open(in_file, 'r') as sample_file:
        data = sample_file.readlines()
    return data


def create_RADAr_samples_with_text_labels(in_file,split="train"):  
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


        

#store text and labels in separate files
def create_RADAr_samples_with_pseudo_labels(in_file,split="train"):  
    print(in_file)
    data = get_data_as_list(in_file)
    context_file = open(f"{folder}/{split}.src","w")
    target_file = open(f"{folder}/{split}.tgt","w")

    for line in data:
        dic = eval(line)# convert line into dictionary
        
        # strip, lower case and truncate text to 512 words
        context = dic['src'].strip().lower()
        context_file.write(context+"\n")
        
        #strip and lower case labels
        if split == "train":
            #dic['tgt'] are lists of the form [['[A_102]'], ['[A_98]'], ['[A_99]'], [], []]
            labels= ""
            for mini_list in dic['tgt']:
                labels += ' '.join([element.strip().lower() for element in mini_list]) 
                #add space afte adding each list to keep the labels separated by space
                labels +=" "
        else:
            #dic['tgt'] are strings of the form  '[A_70] [A_89]'
            labels = dic['tgt'].lower() 
        
        #write label to the target file
        print(labels.strip())
        target_file.write(labels.strip()+"\n") #strip here to remove space at the end
        
                    
    context_file.close()
    target_file.close()
                
def create_sorted_topic_list():
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

    
    
folder = "data/rcv1/" #wos or rcv1-v2
create_RADAr_samples_with_pseudo_labels(folder+"rcv1_train_all_generated_tl.json",split="train")
create_RADAr_samples_with_pseudo_labels(folder+"rcv1_test_all_generated.json",split="test")
create_RADAr_samples_with_pseudo_labels(folder+"rcv1_val_all_generated.json",split="valid")
               
    
    
#create sorted topic file
#create_sorted_topic_list()




    
  
