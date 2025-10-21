# by 
# 2024.3.25
# ref: 

import csv
import json
import os
import re
import sys
import argparse
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
import random
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data, DataLoader, Dataset
from transformers import BertConfig

_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, '..'))
sys.path.append(PROJECT_ROOT)


from openai import OpenAI
import openai

client = OpenAI(
      base_url="https://api.openai.net/v1",
      api_key="sk-xxxxxx"
  )



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
        return os.path.join(self.root, 'raw_org')

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




class ORGDataset_Normalized_After_CHATGPT(ORGDataset):
    def __init__(self, root=None,label_path=None,):
        
        super(ORGDataset_Normalized_After_CHATGPT, self).__init__(root, label_path)


    @property
    def processed_dir(self):
        return os.path.join(self.root, 'raw_org2_after_chatgpt')

    def process(self):
        ''' process raw JSON ORGs '''

        senkeys = ['/OpenAction', '/Action', '/JavaScript', '/JS', '/S']
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
                try:
                    insn_list_norm = block['insn_list'][0]
                except:
                    insn_list_norm= ''
                x.append(insn_list_norm)


            flag = 0
            # get vector from gpt
            gpt_out = list()
            max_tokens = 8191

            #  
            #x = [s.replace('\n', '') for s in x]
            queries = []
            for obj in x:
                tokens = len(tokenizer.encode(obj))
                if tokens <= 8191:
                    queries.append(obj)
                else:
                    #  
                    encoded_input = tokenizer.encode(obj)
                    #  
                    truncated_input = encoded_input[:8191]
                    #  
                    truncated_text = tokenizer.decode(truncated_input)
                    queries.append(truncated_text)


            for ind in range(0, len(queries), 500):
                this_queries = x[ind:ind+500]
                try:
                    response = client.embeddings.create(
                        input=this_queries,
                        model="text-embedding-3-small"
                    )
                    for data in response.data:
                        gpt_out.append(data.embedding)
                    flag = 0
                except openai.BadRequestError:
                    print ('openai.BadRequestError, file:{}'.format(raw_path))
                    flag = 1
                    continue
                except openai.NotFoundError:
                    print ('openai.NotFoundError, file:{}'.format(raw_path))
                    flag = 1
                    continue
                except openai.InternalServerError:
                    print ('openai.InternalServerError, file:{}'.format(raw_path))
                    flag = 1
                    continue
            

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
            x = torch.tensor(gpt_out)
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
    contagio_train_base = '/PDFObj2Vec/obj2vec/preprocess_data/train/'
   
    contagio_label_path = '/PDFObj2Vec/dataset/labels.csv'

  
    dataset = ORGDataset_Normalized_After_CHATGPT(root=contagio_train_base, label_path=contagio_label_path)

    print(len(dataset))




if __name__ == '__main__':
    main()