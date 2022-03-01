import transformers
from pathlib import Path
import os
import json
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,
 AutoModelForTokenClassification, AutoConfig)
from transformers import set_seed
""" This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
"""
print('Transformers version',transformers.__version__)
set_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transformers_model_dowloader():
    print("Download model and tokenizer")
    #loading pre-trained model and tokenizer

    config = AutoConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    

    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

    model.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)
   
    return
if __name__== "__main__":
    transformers_model_dowloader()
