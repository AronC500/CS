import os
import json
import random
import tqdm
import csv
import argparse


class BertDatasetCreator(object):
    ''' Create dataset for BERT training '''

    def __init__(self, data_dir, storepath, corpus_files):
        self.data_dir = data_dir
        self.file_iter = tqdm.tqdm(corpus_files)
        self.storepath = storepath

    def norm_insn(self, insn_list):
        insn_list_norm = []
        for insn in insn_list:
            # #  
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
    
    def normalized_format(self, pairs_count=1000):
        # pairs_count 30 -> 1000 by zgd
        store_f = open(self.storepath, 'a+')

        for path in self.file_iter:
            fullpath = os.path.join(self.data_dir, path)
            print(fullpath)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)

                count = 0
                key_list = list(cfg.keys())
                random.shuffle(key_list)
                for id1 in key_list:
                    block = cfg[id1]
                    if len(block['out_edge_list']) == 0:
                        continue

                    # first block
                    first_seq_norm = self.norm_insn(block['insn_list'])
                    # second block
                    id2 = None
                    for out_edge in block['out_edge_list']:
                        if str(out_edge) != id1:
                            id2 = str(out_edge)
                            break
                    if id2 is None or str(id2) not in cfg.keys():
                        continue
                    second_seq_norm = self.norm_insn(cfg[str(id2)]['insn_list'])

                    for ins in first_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\t')
                    for ins in second_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\n')

                    count += 1
                    if count == pairs_count:
                        break

        store_f.close()


def corpus_file_select(org_dir, mode):
    file_list = os.listdir(org_dir)
    random.shuffle(file_list)
    groups = [file_list]
    corpus_files = []

    for group in groups:
        if mode == 'train':
            # train:val by zgd
            corpus_files.extend(group[:int(len(group) * 0.8)])
        elif mode == 'val':
            corpus_files.extend(group[int(len(group) * 0.8):])
    return corpus_files


def corpus_file_select_train_val(train_org_dir, val_org_dir, mode):
    train_file_list = os.listdir(train_org_dir)
    val_file_list = os.listdir(val_org_dir)

    if mode == 'train':
        return train_file_list
    if mode == 'val':
        return val_file_list





def main2():
    parser = argparse.ArgumentParser(description='Generate BERT Data')
    parser.add_argument('-i', required=True, dest='org_dir', action='store', help='Input ORG files dir' )
    parser.add_argument('-o', required=True, dest='out_file', action='store', help='Output BERT data file' )
    args = parser.parse_args()

    # corpus_files_idx = corpus_file_select(args.org_dir, 'train')
    # print (len(corpus_files_idx))
    for mode in ['train', 'val']:
        store_path = args.out_file + "_" + mode
        corpus_files_idx = corpus_file_select(args.org_dir, mode)
        print(len(corpus_files_idx))
        bert_dataset_creator = BertDatasetCreator(args.org_dir, store_path, corpus_files_idx)
        bert_dataset_creator.normalized_format()




def main():
    parser = argparse.ArgumentParser(description='Generate BERT Data')
    parser.add_argument('-it', required=True, dest='train_org_dir', action='store', help='Input train ORG files dir' )
    parser.add_argument('-iv', required=True, dest='val_org_dir', action='store', help='Input val ORG files dir' )
    parser.add_argument('-o', required=True, dest='out_file', action='store', help='Output BERT data file' )
    args = parser.parse_args()

    

    for mode in ['train', 'val']:
        store_path = args.out_file + "_" + mode
        corpus_files_idx = corpus_file_select_train_val(args.train_org_dir, args.val_org_dir, mode)
        print(len(corpus_files_idx))
        if mode == 'train':
            bert_dataset_creator = BertDatasetCreator(args.train_org_dir, store_path, corpus_files_idx)
            bert_dataset_creator.normalized_format()
        if mode == 'val':
            bert_dataset_creator = BertDatasetCreator(args.val_org_dir, store_path, corpus_files_idx)
            bert_dataset_creator.normalized_format()



if __name__ == '__main__':
    main()


