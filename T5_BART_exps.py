'''
This file is used to test T5 and BART with different configurations for the HTC task. They are uses as they are, then they are extended with new labels, finally the generation process is constrained to a specific set of vocabuarly
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, PhrasalConstraint, BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import pdb


LABEL_SEPARATOR = "|"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
# Define your dataset classpy
class MultilabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower().strip()
        #prompt 1: classif:
        #prompt 2"classify and keep label order: " + text
        #prompt 3 "generate labels in the given order: " + text
        #prompt 4: "classify into the given hierarchical labels: " 
        #prompt 5: "hierarchical multilabel classification: " 
        

        output_text = LABEL_SEPARATOR.join(x for x in self.labels[idx])
  	
        #the next line is for T5 model
        #input_text = "generate labels in the given order: "+text 
        input_text = text #this line for Bart 
        return input_text, output_text

#this function loads the model and tokenizer, extend them (optional) and create constrains (optional)
def get_extended_tokenizer_and_model(fileName,model_name,do_extend,with_constraints):
    # Load tokenizer
    #tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=512)
    tokenizer = BartTokenizer.from_pretrained(model_name,model_max_length=512)
    # Initialize the T5 model
    #model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if do_extend: 
        label_map = {}
        print(f"Original Vocab Size: {tokenizer.vocab_size}")
        with open(fileName, 'rb') as f:
            label_map = pickle.load(f)

        labels = list(label_map.keys())
        labels = [item.lower() for item in labels]
        tokenizer.add_tokens(labels)
        print(model.shared.num_embeddings)
        new_size = model.shared.num_embeddings + len(labels)
        model.resize_token_embeddings(new_size)
        print(model.shared.num_embeddings)

    constraints=None
    #constrain for the decoder
    if with_constraints:
        constraints_list = []
        for label in labels:
            constraints_list.extend(tokenizer(label.strip().lower(), add_special_tokens=False).input_ids)

        constraints = [
        PhrasalConstraint(
            constraints_list)
          ]
    
    return model, tokenizer, constraints
  
     

def get_data(split):
    texts, labels = [], []
      
    #file_path = f"../../HBGL/data/nyt/nyt_{split}.json"
    #file_path = f"../../HBGL/data/rcv1/rcv1_{split}_all.json"
    file_path= f"../../baselines/HBGL/data/nyt/nyt_{split}_all.json"

    with open(file_path, 'r') as json_file:
        for line in json_file.readlines():
            # convert line into dictionary
            entry = eval(line)

            # Extract the values of 'doc_token' and 'doc_label' from the dictionary
            doc_token = entry['token'] #['doc_token']#[token']  # y['src']
            doc_label = [str(x).lower().strip() for x in entry['label']] #['doc_label']]#'label']]  # y['tgt']
            texts.append(doc_token)
            labels.append(doc_label)

    return texts, labels


def get_encoder():
    _, train_labels = get_data("train")
    _, valid_labels = get_data("val")
    _, test_labels = get_data("test")

    label_encoder = MultiLabelBinarizer()
    label_encoder.fit([*train_labels, *test_labels, *valid_labels])

    return label_encoder


def train(model,tokenizer):
    # load train data
    train_texts, train_lbl = get_data("train")
    train_dataset = MultilabelDataset(train_texts, train_lbl)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # load valid data
    val_texts, val_lbl = get_data("val")
    val_dataset = MultilabelDataset(val_texts, val_lbl)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    #model = nn.DataParallel(model)
    # Fine-tune the model
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    total_loss = float('inf')

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        valid_loss = 0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch

            inputs = tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = tokenizer.batch_encode_plus(labels, padding=True, truncation=True, return_tensors='pt').to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            labels=labels['input_ids'], decoder_attention_mask=labels['attention_mask'])

            loss = outputs.loss
            train_loss += loss.mean()

            loss.mean().backward()
            optimizer.step()

        ######################
        # Validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                inputs, labels = batch

                inputs = tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, return_tensors='pt').to(
                    device)
                labels = tokenizer.batch_encode_plus(labels, padding=True, truncation=True, return_tensors='pt').to(
                    device)
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                labels=labels['input_ids'], decoder_attention_mask=labels['attention_mask'])

                loss = outputs.loss
                valid_loss += loss.mean()

            avg_valid_loss = valid_loss / len(val_dataloader)
            avg_train_loss = train_loss / len(train_dataloader)
            print("Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(epoch,
                                                                                                       avg_train_loss,
                                                                                                       avg_valid_loss))
        if total_loss > valid_loss:
            total_loss = valid_loss
            torch.save(model.state_dict(), model_checkpoint)


def test(model,tokenizer,model_checkpoint,result_file,constraints):
    #pdb.set_trace()
    # get data
    test_texts, test_labels = get_data("test")
    print(f"Number of test samples is {len(test_texts)}")
    test_dataset = MultilabelDataset(test_texts, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load  model
    #model = T5ForConditionalGeneration.from_pretrained('t5-base')
    #state_dict = torch.load('models/epc10_t5_prompt3')  # Load the saved state_dict

    # If the saved model was trained using DataParallel
    #if 'module' in state_dict:
    #    model = torch.nn.DataParallel(model)

    # Load the state_dict into the model
    #model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(model_checkpoint))

    #device = 'cpu'
    model.to(device)
    model.eval()

    predicted_lbls, original_lbls = [], []
    for batch in tqdm(test_dataloader):
        inputs, labels = batch
        encoded_inputs = tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, return_tensors='pt').to(
            device)
        with torch.no_grad():
            outputs = model.generate(input_ids=encoded_inputs['input_ids'],
                                     attention_mask=encoded_inputs['attention_mask'],
                                     max_length=25,  # Maximum number of labels to generate
                                     num_beams=5, temperature=0.5,  # Beam search width
                                     constraints=constraints,
                                     early_stopping=True)
        decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predicted_lbls.extend(decoded_predictions)
        original_lbls.extend(labels)
    ready_original_labels, ready_prediction_labels = [], []
    # save predictions along with original labels in a text file
    f = open(result_file, "w")
    for p, o in zip(predicted_lbls, original_lbls):
        preds = [e.strip() for e in p.split(LABEL_SEPARATOR)]
        ready_prediction_labels.append(preds)

        origs = [e.strip() for e in o.split(LABEL_SEPARATOR)]
        ready_original_labels.append(origs)

        f.write(f"{preds}\n{origs}\n\n")
    f.close()

    compute_metrics(ready_original_labels, ready_prediction_labels)


# this function takes two lists of labels and compute the matrics on them
def compute_metrics(origin, preds):
    encoder = get_encoder()
    original_lbls, predicted_lbls = [], []

    original_lbls = encoder.transform(origin)
    predicted_lbls = encoder.transform(preds)

    # calculate F1-score
    f1_score_val = f1_score(original_lbls, predicted_lbls, average=None)
    # calculate macro-average f1-score
    macro_f1_score = f1_score(original_lbls, predicted_lbls, average='macro')

    micro_f1_score = f1_score(original_lbls, predicted_lbls, average='micro')
    print(f"f1 score:{f1_score_val}\nmacro f1-score:{macro_f1_score}\nmicro_f1_score:{micro_f1_score}\n\n")


# this function reads the output file and returns two lists
def read_labels_file(file_name):
    preds, orgins = [], []
    f = open(file_name, 'r')
    i=0
    for line in tqdm(f.readlines()):
        if line == '\n':
            i=0
            continue
        lst = eval(line)
        if not isinstance(lst,list):
            print("some problem")
        labels = lst[0].split(LABEL_SEPARATOR)
        if i == 0:
            preds.append(labels)
            i=1
        else:
            orgins.append(labels)
    return orgins, preds
#configurations
do_extend = False #extend model and tokenizer with new vocabulary
with_constraints = False # create constrains to force the model to generate from specific vocabulary
result_file = 'results/bart_on_nyt.txt'
model_checkpoint = 'models/bart_on_nyt.pt' #save 
model_name = 'facebook/bart-base'
#'t5-base' #'facebook/bart-base'

#extend the tokenizer and model and create constraints if they exist
model,tokenizer, constraints= get_extended_tokenizer_and_model('../../baselines/HBGL/data/nyt/label_map.pkl',model_name,do_extend,with_constraints)
train(model,tokenizer)
test(model,tokenizer,model_checkpoint,result_file,constraints)

