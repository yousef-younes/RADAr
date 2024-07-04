## RADAr: A Transformer-based Autoregressive Decoder Architecture for Hierarchical Text Classification

This repo contains the code for The RADAr model. RADAR is a sequence-to-sequence model for Hierarchical Text Classification. 

## Modules Description
Here is a brief description about the funcationality of the different modules in the repo.

1. We use the same data splits and  preprocessing as HBGL. So first, the preprocessing done for [HBGL](https://github.com/kongds/HBGL) must be run on each dataset. 
2. The prepare\_data.py module from the data\_preparation directory should be used. As a result, each sample will be split into text and lable and six files will be created. train.src, train.tgt, val.src, val.tgt, test.src, test.tgt. The .src files contains the text while .tgt files contains the labels.
3. To get the data statistics, the datasets\_statistics notebook can be used. 
4. The organize\_labels\_level\_wise.py module seperates the labels level-wise.
5. The organize\_labels\_path\_wise.py" module organized the labels in seperate paths.
6. The sweep\_code is used for hyperparameter optimization.
7. The T5\_BART\_exps.py is the module used to get the resutls using T5 and BART model. 
8. The "analyze\_resutls" notebook is used to analyze the model resutls.

## To Run the code:
To run the 
- After preparing the data using the data preparation modules. 
- Specify the data, log, model and results directories and other parameters in the yaml file corresponsing to each dataset. 
- To train and test the model, please use the comman:  python main.py xxx

where "xxx" is a three letter acronym for the dataset. It is either wos, nyt, or rcv indicating the three datasets WOS, NTY and RCV1-V2 respectively. 





