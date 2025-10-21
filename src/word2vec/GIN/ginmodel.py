#################################
##### by  2024.3.17######
#################################

from torch_geometric.data import Data
from dgl.data import DGLDataset
import os
import copy
import torch
from torch import nn
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgl.nn import GraphConv, GINConv
import numpy as np
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv, GINConv

def pyg_to_dgl(pyg_graph):
   
    dgl_graph = dgl.graph((pyg_graph.edge_index[0], pyg_graph.edge_index[1]))
   
    dgl_graph.ndata['x'] = pyg_graph.x
   
    dgl_graph.edata['edge_attr'] = pyg_graph.edge_attr
   
    return dgl_graph, pyg_graph.y



class OrgDGLDataset(DGLDataset):
    def __init__(self, root):
        self.root = root
        super(OrgDGLDataset, self).__init__(name='org_dgl_dataset')
        self.load()  #

    def load(self):
        # 
        self.graphs = []
        self.labels = []
        for filename in os.listdir(self.root):
            if filename.endswith('.pt') and filename != 'pre_filter.pt' and filename != 'pre_transform.pt': # by zgd
                file_path = os.path.join(self.root, filename)

                #debug
                # print (file_path)
                try:
                    pyg_graph = torch.load(file_path)
                    
                   
                    if pyg_graph.edge_index.size(0) > 0:
                        src, dst = pyg_graph.edge_index
                    else:
                        src, dst = [], []

                    
                    dgl_graph = dgl.graph((src, dst), num_nodes=pyg_graph.x.shape[0])
                    if hasattr(pyg_graph, 'x'):
                        dgl_graph.ndata['feat'] = pyg_graph.x
                    if hasattr(pyg_graph, 'edge_attr'):
                        dgl_graph.edata['feat'] = pyg_graph.edge_attr
                    
                    self.graphs.append(dgl_graph)
                    self.labels.append(pyg_graph.y)
                except:
                    print ('【ERROR】', file_path)
                    

    def __getitem__(self, idx):
       
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
       
        return len(self.graphs)

def load_pdf_org_dataset(dataset_dir):
    dataset = OrgDGLDataset(root=dataset_dir)

    feature_dim = dataset[0][0].ndata['feat'].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    num_classes = torch.max(labels).item() + 1
    
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    # print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)

def collate(samples):
   
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels) 
    return batched_graph, batched_labels



class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(in_feats, hidden_feats), 'mean')
        self.conv2 = GINConv(nn.Linear(hidden_feats, hidden_feats), 'mean')
        #self.conv3 = GINConv(nn.Linear(hidden_feats, hidden_feats), 'mean')
        self.classify = nn.Linear(hidden_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        #h = F.relu(h)
        #h = self.conv3(g, h)
        g.ndata['h'] = h
        hg = dgl.max_nodes(g, 'h')
        return self.classify(hg)
    



class GINAutoencoder(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GINAutoencoder, self).__init__()
        self.encoder1 = GINConv(nn.Linear(in_feats, hidden_size), 'mean')
        self.encoder2 = GINConv(nn.Linear(hidden_size, hidden_size), 'mean')
        self.decoder1 = nn.Linear(hidden_size, hidden_size)
        self.decoder2 = nn.Linear(hidden_size, out_feats)
    
    def forward(self, g, features):
        h = F.relu(self.encoder1(g, features))
        h = F.relu(self.encoder2(g, h))
        g.ndata['h'] = h  
        hg = dgl.mean_nodes(g, 'h')  
        reconstructed = F.relu(self.decoder1(hg))
        reconstructed = self.decoder2(reconstructed)
        return hg, reconstructed

class GraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphClassifier, self).__init__()
        self.autoencoder = GINAutoencoder(in_feats, hidden_size, in_feats)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)  
    
    def forward(self, g, features):
        encoded, _ = self.autoencoder(g, features)
        global_feats = dgl.max_nodes(g, 'h')
        out = self.classifier(global_feats)
        return out  



def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return acc, recall, precision, f1, tpr, tnr










def train_baseline(model, device, data_loader, adv_loader, epochs, records_dir, log_file):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    records_dir = records_dir
    if not os.path.exists(records_dir):
        os.mkdir(records_dir)
    record_fd = open(os.path.join(records_dir, log_file), 'w+')
    model.train()
    for epoch in range(epochs):
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata['feat'].to(device)
            edge_weight = batched_graph.edata['feat']
            logits = model(batched_graph, features).to(device)
            labels = labels.to(device).squeeze()  
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      
        acc, recall, precision, f1, fpr = evaluate(model, device)
       

        print(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}')
        model_dict = copy.deepcopy(model.state_dict())
        fp = os.path.join(records_dir, '{}.pth'.format(epoch))
        record_fd.write(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f} \n')
        torch.save(model_dict, fp)
    record_fd.close()





   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GIN Training')
    parser.add_argument('-tr', required=True, dest='train_dir', action='store', help='Input train dataset' )
    parser.add_argument('-te', required=True, dest='test_dir', action='store', help='Input test dataset' )
    parser.add_argument('-r', required=True, dest='record_dir', action='store', help='record_dir' )
    parser.add_argument('-o', required=True, dest='log_file', action='store', help='output log' )
    parser.add_argument('-id', required=True, dest='device_id', action='store', help='device id' )
    args = parser.parse_args()

    train_dataset = args.train_dir
    test_dataset = args.test_dir
    log_file = args.log_file    
    record_dir = args.record_dir
    deviceid = args.device_id



    train_dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(train_dataset)
    test_dataset, (feature_dim3,feature_dim4) = load_pdf_org_dataset(test_dataset)
   

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate)


    
    device = torch.device(f"cuda:{deviceid}")
    # 初始化模型、优化器和损失函数
    model = GIN(in_feats=512, hidden_feats=256, num_classes=2).to(device)


    epochs = 50
   
    train_baseline(model, device, train_loader, test_loader, epochs, record_dir,log_file)