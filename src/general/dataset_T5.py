# by 
# 2024.3.26
# ref: 

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download


import csv
import json

import re
import sys
import argparse
import random
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data, DataLoader, Dataset
from transformers import BertConfig

_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, '..'))
sys.path.append(PROJECT_ROOT)




# from transformers import BertTokenizer, BertModel

from transformers import AutoModel, AutoTokenizer
checkpoint = "Salesforce/codet5p-110m-embedding"
# Salesforce/codet5p-110m-embedding

class ORGDataset(Dataset):
    def __init__(self, root, label_path):
        self.number_of_classes = 2
        self.label_path = label_path

        # labels
        self.labels = {}
        with open(self.label_path, 'r') as f:
            data = csv.reader(f)
            for row in data:
                if row[0] == 'Id':
                    continue
                self.labels[row[0]] = int(row[1])

        super(ORGDataset, self).__init__(root, transform=None, pre_transform=None)

    @property
    def num_classes(self):
        return self.number_of_classes

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'org')

    @property
    def raw_file_names(self):
        files_names = os.listdir(self.raw_dir)
        # exclude .DS_Store
        if '.DS_Store' in files_names:
            files_names.remove('.DS_Store')
        return files_names

    @property
    def processed_file_names(self):
        file_names = ['data_{}_{}.pt'.format(
            i, filename.split('.')[0]) for i, filename in enumerate(self.raw_file_names)]
        return file_names

    def raw_file_name_lookup(self, idx):
        return self.raw_file_names[idx]

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        raise NotImplementedError

    def get(self, idx):
        raw_file_name = self.raw_file_name_lookup(idx).split('.')[0]
        data = torch.load(os.path.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, raw_file_name)))
        return data





def batch_bert_out(model, tokenizer, device, texts, batch=128):
    features = []
    for idx in range(0, len(texts), batch):
        batch_texts = texts[idx:idx+batch]
        encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
       
        input_ids = encoded_batch['input_ids'].to(device)
        # attention_mask = encoded_batch['attention_mask'].to(device)
        
        # with torch.no_grad():
        #     outputs = model(input_ids, attention_mask)
        #     last_hidden_states = outputs.last_hidden_state
        # features += last_hidden_states[:, 0, :].to('cpu').numpy().tolist()
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        features += last_hidden_states.to('cpu').numpy().tolist()
    return np.array(features)


class ORGDataset_Normalized_After_BASEBERT(ORGDataset):
    def __init__(self, root=None,label_path=None,):
        
        super(ORGDataset_Normalized_After_BASEBERT, self).__init__(root, label_path)


    @property
    def processed_dir(self):
        return os.path.join(self.root, 'org_after_codeT5')

    def process(self):

       
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

        device = torch.device("cuda:0")
        model.to(device)



        ''' process raw JSON ORGs '''

        senkeys = []
        idx = 0
        print ('raw files counts: ', len(self.raw_file_names))
        for raw_path in tqdm.tqdm(self.raw_file_names):
            
            #save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            topath = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            if topath in os.listdir(self.processed_dir):
                idx += 1
                continue

            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            ## y (label)
            y = int(self.labels[raw_path])

            addr_to_id = dict()  # {str: int}
            current_node_id = -1

            x = list()  # node attributes
            
            for addr, block in cfg.items():  # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id
                # node process
                insn_list_norm = ''
                sentence = []
                row = ''
                for ir in block['insn_list']:
                    row = ','.join(ir)
                    sentence.append(row)
                insn_list_norm = '. '.join(sentence)
                x.append(insn_list_norm)

            flag = 0
            # get vector from gpt


            #basebert_out = list()  
            basebert_out = batch_bert_out(model, tokenizer, device, x, batch=128).tolist()
            #batch_bert_out(model, tokenizer, device, texts, batch=128):
           

                # get sparse adjacent matrix
            edge_index = list()
            edge_attr = list()
            for addr, block in cfg.items():  # addr is `str`
                start_nid = addr_to_id[addr]
                isins = [num for elem in block['insn_list'] for num in elem]
                for out_edge in block['out_edge_list']:
                    if str(out_edge) in addr_to_id.keys():
                        end_nid = addr_to_id[str(out_edge)]
                        ## edge_index
                        edge_index.append([start_nid, end_nid])
                        intersection = set(senkeys) & set(isins)
                        if intersection:
                            edge_attr.append(1)
                        else:
                            edge_attr.append(0)

            # Data
            x = torch.tensor(basebert_out)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            # save
            assert (self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            if flag == 0:
                torch.save(data, save_path)

            idx += 1




def main():
    root = '../preprocess_data/extended'

  


    label_path = '../../dataset/_pdfs_labels.csv'
    # by zgd
    dataset = ORGDataset_Normalized_After_BASEBERT(root=root, label_path=label_path)

    print(len(dataset))




if __name__ == '__main__':
    main()