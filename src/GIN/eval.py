from ginmodel import load_pdf_org_dataset,  evaluate
import torch
import os
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import dgl
import torch.nn.functional as F
from dgl.nn import GraphConv, GINConv
import argparse


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
setup_seed(42)

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




def evaluate_mal(model, data_loader, device):
    model.eval()  
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())
    yp = np.array(y_pred)
    # print (len(yp[yp==1]), '/', len(yp))
    return (len(yp[yp==1])), len(yp)


def evaluate_ben(model, data_loader, device):
    model.eval()  
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())
    yp = np.array(y_pred)
    return (len(yp[yp==0])), len(yp)


def calculate_metrics(extended_mal, extended_ben):
   
    TP, P = extended_mal  
    TN, N = extended_ben  
    FP = N - TN  
    FN = P - TP  
    accuracy = (TP + TN) / (P + N)  
    recall = TP / P if P > 0 else 0  
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  
    tpr = recall  
    tnr = TN / N if N > 0 else 0  

    return accuracy, recall, precision, f1, tpr, tnr


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-mode', required=True, choices=['baseline', 'extended', 'adv'], help='Mode selection')
    parser.add_argument('-dataset', required=True, nargs='+', help='Input eval dataset directories')
    parser.add_argument('-ginmodel', required=True, help='GIN model path')

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Datasets: {args.dataset}")  # 

  
    dataset = args.dataset
    ginmodel_path = args.ginmodel
    mode = args.mode
    
    deviceid = 0  
    device = torch.device(f"cuda:{deviceid}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    

    
    if mode == 'baseline':  
        dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(dataset[0])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
        model = GraphClassifier(in_feats=1024, hidden_size=256, num_classes=2).to(device)
        model.load_state_dict(torch.load(ginmodel_path))
        model.to(device)
        acc, recall, precision, f1, tpr, tnr = evaluate(model, data_loader, device)
        print ('Basline evaluation results:')
        print(f'Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | TPR: {tpr:.4f} | TNR: {tnr:.4f}')
    if mode == 'extended':
        for dataset_path in dataset:
            if 'ben' in dataset_path:
                dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(dataset_path)
                data_loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
                model = GraphClassifier(in_feats=1024, hidden_size=256, num_classes=2).to(device)
                model.load_state_dict(torch.load(ginmodel_path))
                model.to(device)
                extended_ben = evaluate_ben(model, data_loader, device)
            if 'mal' in dataset_path:
                dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(dataset_path)
                data_loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
                model = GraphClassifier(in_feats=1024, hidden_size=256, num_classes=2).to(device)
                model.load_state_dict(torch.load(ginmodel_path))
                model.to(device)
                extended_mal = evaluate_mal(model, data_loader, device)
        acc, recall, precision, f1, tpr, tnr = calculate_metrics(extended_mal, extended_ben)
        print ('Extended evaluation results:')
        print(f'Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | TPR: {tpr:.4f} | TNR: {tnr:.4f}')
    if mode == 'adv':
        dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(dataset[0])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
        model = GraphClassifier(in_feats=1024, hidden_size=256, num_classes=2).to(device)
        model.load_state_dict(torch.load(ginmodel_path))
        model.to(device)
        tp, P = evaluate_mal(model, data_loader, device)
        tpr = tp/P
        print ('Adv evaluation results:')
        print(f'TPR: {tpr:.4f} ')

      # python eval.py -mode extended -dataset ./preprocess_data/mal0406/org_after_prebert_contagio++_hidden1024/ ./preprocess_data/allben_pred0_by_previous/org_after_prebert_contagio++_hidden1024/ -ginmodel ./GIN-BERT65k.pth
      # python eval.py -mode baseline -dataset ./preprocess_data/reverse_mimicry_wine08/org_after_prebert_contagio++_hidden1024/  -ginmodel ./GIN-BERT65k.pth
      # python eval.py -mode adv -dataset ./preprocess_data/reverse_mimicry_wine08/org_after_prebert_contagio++_hidden1024/  -ginmodel ./GIN-BERT65k.pthe