# by 
# 2024.2.2
# ref: 

import csv
import json
import os
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

from bert.model import PREBERT
from bert.vocab import *
from utils.set_random import setup_seed

# set random seed
setup_seed(42)

sys.path.append('../dataset/')




class BBSDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, segment_labels, attention_mask):
        self.inputs = inputs  # (total, seq_len)
        self.segment_labels = segment_labels  # (total, seq_len)
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ = self.inputs[idx]
        segment_label_ = self.segment_labels[idx]
        attention_mask_ = self.attention_mask[idx]

        return {
            'inputs': torch.tensor(input_),
            'segment_labels': torch.tensor(segment_label_),
            'attention_mask': torch.tensor(attention_mask_)
        }


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

    def train_val_split(self, k):
        ''' k fold train_val_split 
        k = 0, 1, 2, 3, 4
        '''
        # TODO: 5-fold
        # 0 0.2 0.4 0.6 0.8
        label_list = list()
        for filename in self.raw_file_names:
            label = self.labels[filename]
            label_list.append(int(label))

        groups = [[] for _ in range(2)]
        for i, label in enumerate(label_list):
            groups[label - 1].append(i)

        train_idx = [];
        val_idx = []

        for group in groups:
            group_len = len(group)
            slice_1 = int(group_len * 0.2 * k)
            slice_2 = int(group_len * 0.2 * (k + 1))

            train_idx.extend(group[0:slice_1])
            train_idx.extend(group[slice_2:])

            val_idx.extend(group[slice_1:slice_2])
            # train_idx.extend(group[int(len(group) * 0.15):])
            # val_idx.extend(group[:int(len(group) * 0.15)])

        return train_idx, val_idx

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        raise NotImplementedError

    def get(self, idx):
        raw_file_name = self.raw_file_name_lookup(idx).split('.')[0]
        data = torch.load(os.path.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, raw_file_name)))
        return data




class ORGDataset_Normalized_After_BERT(ORGDataset):
    def __init__(self, root=None, hidden_size=512, label_path=None,
                 vocab_path=None, bert_path=None, seq_len=None):
        
        self.vocab_path = vocab_path
        self.bert_path = bert_path
        self.seq_len = seq_len if seq_len else 64
        self.hidden_size = hidden_size
        #self.outdir = processed_dir
        #self.label_path = label_path

        super(ORGDataset_Normalized_After_BERT, self).__init__(root, label_path)

    def norm_insn(self, insn_list):
        insn_list_norm = []
        for insn in insn_list:
            # # 1
            # if insn[1] == 'STREAM':
            #     insn_norm = insn[1]
            # elif insn[1] == 'NAME' or insn[1] == 'BOOL':
            #     insn_norm = insn[0] + '_' + insn[1] + '_' + insn[2]
            # else:
            #     insn_norm = insn[0] + '_' + insn[1]

            # 
            if insn[1] == 'STREAM':
                insn_norm = insn[1]
            else:
                insn_norm = insn[0] + '_' + insn[1]

            insn_list_norm.append(insn_norm)
        return insn_list_norm

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'org_after_prebert')

    def process(self):
        ''' process raw JSON ORGs '''

        # vocab
        self.vocab = WordVocab.load_vocab(self.vocab_path)

        # `PREBERT`
        config = BertConfig(hidden_size=self.hidden_size,
                            vocab_size=len(self.vocab), num_attention_heads=8,
                            num_hidden_layers=8, return_dict=True)
        self.model = PREBERT(config)  # PREBERT

        device_flag = 'cuda' #default
        if torch.backends.mps.is_available():
            device = torch.device('mps:0')
            self.model.load_state_dict(torch.load(self.bert_path, map_location=torch.device('mps')))
            self.bert = self.model.bert.bert
            self.bert = self.bert.to(device)
            self.bert.eval()
            device_flag = 'mps'
        else:
            self.model.load_state_dict(torch.load(self.bert_path, map_location=torch.device('cuda')))
            self.bert = self.model.bert.bert
            self.bert = self.bert.cuda()
            self.bert.eval()
        print('Loaded BERT model.')

        # self.bert = self.model.bert.bert
        # self.bert = self.bert.cuda()
        # self.bert.eval()
        senkeys = ['/OpenAction', '/Action', '/JavaScript', '/JS', '/S']
        idx = 0
        print ('raw files counts: ', len(self.raw_file_names))
        for raw_path in tqdm.tqdm(self.raw_file_names):
            

            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            ## y (label)
            y = int(self.labels[raw_path])

            addr_to_id = dict()  # {str: int}
            current_node_id = -1

            x = list()  # node attributes
            seg = list()  # node segment labels
            mask = list()  # attention mask
            for addr, block in cfg.items():  # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                # get tokenized opcode sequence as node attributes
                tokenized_seq = []
                seq_norm = self.norm_insn(block['insn_list'])
                for insn in seq_norm:

                    tokenized = self.vocab.stoi.get(insn, self.vocab.unk_index)
                    tokenized_seq.append(tokenized)

                # add [CLS] and [SEP]
                tokenized_seq = [self.vocab.sos_index] + tokenized_seq + [self.vocab.eos_index]

                # max seq len
                tokenized_seq = tokenized_seq[:self.seq_len]
                segment_label = [0 for _ in range(len(tokenized_seq))][:self.seq_len]
                attention_mask = [1 for _ in range(len(tokenized_seq))][:self.seq_len]

                # padding
                padding = [self.vocab.pad_index for _ in range(self.seq_len - len(tokenized_seq))]
                tokenized_seq.extend(padding)
                segment_label.extend(padding)
                attention_mask.extend(padding)

                x.append(tokenized_seq)
                seg.append(segment_label)
                mask.append(attention_mask)

            # construct dataset
            bbs_dataset = BBSDataset(x, seg, mask)
            bbs_dataloader = torch.utils.data.DataLoader(bbs_dataset, batch_size=100, shuffle=False, num_workers=0)

            bert_out = list()
            for bbs_data in bbs_dataloader:
                if device_flag == 'cuda':
                    outputs = self.bert(input_ids=bbs_data['inputs'].cuda(), 
                                        attention_mask=bbs_data['attention_mask'].cuda(), 
                                        token_type_ids=bbs_data['segment_labels'].cuda(),
                                        return_dict=True)
                if device_flag == 'mps':
                    outputs = self.bert(input_ids=bbs_data['inputs'].to(device), 
                                        attention_mask=bbs_data['attention_mask'].to(device), 
                                        token_type_ids=bbs_data['segment_labels'].to(device),
                                        return_dict=True)

                bert_out.extend(outputs.pooler_output.tolist())

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
            x = torch.tensor(bert_out)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            # save
            assert (self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1

def main():
    parser = argparse.ArgumentParser(description='Generate torch geometric data')
    parser.add_argument('-r', required=True, dest='base_dir', action='store', help='Base dir' )
    parser.add_argument('-v', required=True, dest='vocab_path', action='store', help='Input vocab file path' )
    parser.add_argument('-b', required=True, dest='bert_path', action='store', help='Input BERT model path' )
    parser.add_argument('-label', required=True, dest='label_file', action='store', help='Input label file path' )
    #parser.add_argument('-o', required=True, dest='out_dir', action='store', help='Output ORG after prebert files dir')
    
    args = parser.parse_args()

    base_dir = args.base_dir
    vocab_path = args.vocab_path
    bert_path = args.bert_path
    label_path = args.label_file

    # by zgd
    dataset = ORGDataset_Normalized_After_BERT(root=base_dir, vocab_path=vocab_path, bert_path=bert_path, hidden_size=512, label_path=label_path, seq_len=64)

    print(len(dataset))




if __name__ == '__main__':
    main()



