import os
import json
from transformers import BertTokenizer, BertForPreTraining, BertConfig
import torch
#  
corpus_directory = '/data/PDFObj2Vec/obj2vec/preprocess_data/usenix24_train/raw_org2'  
text_data = []
import tqdm
#  
for filename in tqdm.tqdm(os.listdir(corpus_directory)):
    file_path = os.path.join(corpus_directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for block in data.values():
            insn_list = block['insn_list']
            text_data.extend(insn_list)  #  
#  



device = torch.device("cuda:0")

 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


 
tokenized_texts = [tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, truncation=True, padding='max_length') for text in text_data]
 
import torch

input_ids = torch.tensor([x['input_ids'] for x in tokenized_texts])
attention_masks = torch.tensor([x['attention_mask'] for x in tokenized_texts])

 
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)

 
torch.save(dataset, 'raworg2_retrained_bert_dataset.pt')

