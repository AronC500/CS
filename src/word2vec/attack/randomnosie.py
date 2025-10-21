import torch as th
import torch.nn.functional as F
import dgl
import os, sys
from copy import deepcopy
import random
from dgl.data import citation_graph as citegrh
import networkx as nx
import torch
import numpy as np
from easydict import EasyDict
from torch.nn.functional import normalize
from dgl import DGLGraph
from networkx.generators.random_graphs import fast_gnp_random_graph
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import betweenness_centrality as betweenness
from numpy.random import multivariate_normal
from dgl.data import DGLDataset
import argparse


from tqdm import tqdm

from grabnel.attack.base_attack import BaseAttack
from grabnel.attack.utils import classification_loss,correct_predictions, population_graphs, random_sample_flip, random_sample_rewire_swap, get_allowed_nodes_k_hop, extrapolate_breakeven, nettack_loss, number_of_correct_predictions
from grabnel.attack.utils import number_of_correct_predictions, setseed, classification_loss #
from grabnel.models.utils import get_model_class #
from torch.utils.data import DataLoader, default_collate #
import torch.optim as optim #

class OrgDGLDataset_v2(DGLDataset):
    def __init__(self, root):
        self.root = root
        super(OrgDGLDataset_v2, self).__init__(name='org_dgl_dataset')
        self.graphs = []
        self.labels = []
        self.load()  

    def load(self):
       
        for filename in os.listdir(self.root):
            if filename.endswith('.pt') and filename not in ['pre_filter.pt', 'pre_transform.pt']:  
                file_path = os.path.join(self.root, filename)
                pyg_graph = torch.load(file_path)

               
                src, dst = pyg_graph.edge_index if pyg_graph.edge_index.size(0) > 0 else ([], [])
                
                try:
                    dgl_graph = dgl.graph((src, dst), num_nodes=pyg_graph.x.shape[0])

                   
                    if hasattr(pyg_graph, 'x'):
                        
                        dgl_graph.ndata['node_attr'] = pyg_graph.x
                        dgl_graph.ndata['node_attr'] = dgl_graph.ndata['node_attr'].float() # float32
                    if hasattr(pyg_graph, 'edge_attr'):
                        dgl_graph.edata['edge_attr'] = pyg_graph.edge_attr
                
                   
                    self.graphs.append(dgl_graph)
                    self.labels.append(int(pyg_graph.y.item()))  
                except:
                    print('error:{}'.format(filename))
                    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
    

def load_pdf_org_dataset_v2(dataset_dir):
    dataset = OrgDGLDataset_v2(root=dataset_dir)

    feature_dim = dataset[0][0].ndata['node_attr'].shape[1] # node_attr
    labels = torch.tensor([x[1] for x in dataset])
    num_classes = torch.max(labels).item() + 1

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels)  
    return batched_graph, batched_labels


def train_GIN_model(train_dir, test_dir, device):
    
    train_dataset, (_1, _2) = load_pdf_org_dataset_v2(train_dir)
    test_dataset, (_1, _2) = load_pdf_org_dataset_v2(test_dir)
    print(f"train_dataset len is {len(train_dataset)}, test_dataset len is {len(test_dataset)}")
    
    lr = 0.001
    wd = 0.0001
    num_epochs = 50
    seed = 42
    
    setseed(seed)
    feature_dim = train_dataset[0][0].ndata['node_attr'].shape[1]
    number_of_labels = 2
    is_binary = 2
    
    model_class = get_model_class("gin")
    model = model_class(feature_dim, number_of_labels)
    # device = torch.device(f"cuda:{device}")
    model = model.to(device)
    
    # specify loss function
    loss_fn = classification_loss(is_binary)
    
    # train model
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.
    best_model = None
    # training_logs = []
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    
        # training step
        model.train()
        train_loss, train_acc = 0, 0
        for i, (graphs, labels) in enumerate(train_loader):
            graphs, labels = graphs.to(device), labels.to(device)
            labels = labels.long()
            predictions = model(graphs)
            # GIN models still give a bug here:
            if is_binary and predictions.shape[1] > 1:
                predictions = predictions[:, 0]
    
            loss = loss_fn(predictions, labels).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += number_of_correct_predictions(predictions, labels, is_binary).detach().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
    
        # evaluation step
        model.eval()
        valid_loss, valid_acc = 0, 0
        with torch.no_grad():
            for i, (graphs, labels) in enumerate(test_loader):
                graphs, labels = graphs.to(device), labels.to(device)
                labels = labels.long()
                predictions = model(graphs)
                if is_binary and predictions.shape[1] > 1:
                    predictions = predictions[:, 0]
    
                loss = loss_fn(predictions, labels).mean()
                valid_loss += loss.detach().item()
                valid_acc += number_of_correct_predictions(predictions, labels, is_binary).detach().item()
            valid_loss /= len(test_loader)
            valid_acc /= len(test_loader.dataset)
    
        # save best model
        if valid_acc > best_val_acc:
            print('Best val acc recorded at epoch ', epoch)
            best_model = deepcopy(model)
            best_val_acc = valid_acc
    
        print(f"epoch: {epoch}, train_loss: {'{:.4f}'.format(train_loss)}, \
                valid_loss: {'{:.4f}'.format(valid_loss)}, train_acc: {'{:.2f}'.format(train_acc)}\
                valid_acc: {'{:.2f}'.format(valid_acc)}", flush=True)
    
    return best_model


def save_adversarial_example(filename, adv_graph, save_dir):
    adv_filename = "_".join(["adv"] + filename.split("_")[1:])
    adv_filepath = os.path.join(save_dir, adv_filename)  
    torch.save(adv_graph, adv_filepath)

def getScore(K, data):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    return Random.sum(dim=0)


def getScoreGreedy(K, data, bar, num, beta):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    W = th.zeros(data.size, data.size)
    for i in range(data.size):
        value, index = th.topk(Random[i], beta)
        for j, ind in zip(value, index):
            if j != 0:
                W[i, ind] = 1
    SCORE = W.sum(dim=0)
    ind = []
    l = [i for i in range(data.size) if data.g.out_degree(i) <= bar]
    for _ in range(num):
        cand = [(SCORE[i], i) for i in l]
        best = max(cand)[1]
        for neighbor in data.g.out_edges(best)[1]:
            if neighbor in l:
                l.remove(neighbor)
        ind.append(best)
        for i in l:
            W[:, i] -= (W[:, best] > 0) * 1.0
        SCORE = th.sum(W > 0, dim=0)
    return np.array(ind)


def getThrehold(g, size, threshold, num):
    degree = g.out_degrees(range(size))
    Cand_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
    threshold = int(size * threshold)
    bar, _ = Cand_degree[threshold]
    Baseline_Degree = []
    index = [j for i, j in Cand_degree if i == bar]
    if len(index) >= num:
        Baseline_Degree = np.array(index)[np.random.choice(len(index),
                                                           num,
                                                           replace=False)]
    else:
        while 1:
            bar -= 1
            index_ = [j for i, j in Cand_degree if i == bar]
            if len(index) + len(index_) >= num:
                break
            for i in index_:
                index.append(i)
        for i in np.array(index_)[np.random.choice(len(index_),
                                                   num - len(index),
                                                   replace=False)]:
            index.append(i)
        Baseline_Degree = np.array(index)
    random = [j for i, j in Cand_degree if i <= bar]
    Baseline_Random = np.array(random)[np.random.choice(len(random),
                                                        num,
                                                        replace=False)]
    return bar, Baseline_Degree, Baseline_Random


def getIndex(g, Cand, bar, num):
    ind = []
    for j, i in Cand:
        if g.out_degree(i) <= bar:
            ind.append(i)
        if len(ind) == num:
            break
    return np.array(ind)


def synthetic_data(num_node=3000, num_feature=10, num_class=2, num_important=4):
    gnp = nx.barabasi_albert_graph(num_node, 2)
    gnp.remove_edges_from(nx.selfloop_edges(gnp))
    g = DGLGraph(gnp)
    g.add_edges(g.nodes(), g.nodes())
    data = EasyDict()
    data.graph = gnp
    data.num_labels = num_class
    data.g = g
    data.adj = g.adjacency_matrix().to_dense()
    means = np.zeros(num_node)
    degree = np.zeros((num_node, num_node))
    for i in range(num_node):
        degree[i,i] = data.adj[i].sum()**-0.5
    lap_matrix = np.identity(num_node) - np.matmul(np.matmul(degree, data.adj.numpy()), degree)
    cov = np.linalg.inv(lap_matrix + np.identity(num_node))
    data.features = th.from_numpy(multivariate_normal(means, cov, num_feature).transpose())
    data.features = data.features.float().abs()
    g.ndata['x'] = data.features
    W = th.randn(num_feature) * 0.1
    W[range(num_important)] = th.Tensor([10,-10,10,-10])
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1)
    logits = th.sigmoid(th.matmul(th.matmul(normalize(data.adj, p=1, dim=1), data.features), W)) 
    labels = th.zeros(num_node)
    labels[logits>0.5] = 1      
    data.labels = labels.long()
    data.size = num_node
    return data


def load_data2(dgl_graph, num_class=2):
    data = EasyDict()
    data.graph = dgl_graph.to_networkx()
    data.num_labels = num_class
    data.g = dgl_graph
    data.adj = dgl_graph.adjacency_matrix().to_dense()

    

    data = synthetic_data()
    print("============data %s===============")
    print (type(data),data)
    data.features = th.FloatTensor(data.features)
    data.labels = th.LongTensor(data.labels)
    data.size = data.labels.shape[0]
    print("============data %s===============")
    print (data.features,data.features.shape)
    
    print (data.labels)
    print (data.size)
    g = data.graph
    print (g)
    g.remove_edges_from(nx.selfloop_edges(g))
    print("============data %s===============")
    g = DGLGraph(g)
    print (g)
    g.add_edges(g.nodes(), g.nodes())
    data.g = g
    data.adj = g.adjacency_matrix().to_dense()
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1)
    print("============Successfully Load %s===============")
    print (data)
    return data


    
    


def load_attack_data(data_dir):
    attack_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pt') and filename not in ['pre_filter.pt', 'pre_transform.pt']:
            filepath = os.path.join(data_dir, filename)
            pyg_graph = torch.load(filepath)

           
            src, dst = pyg_graph.edge_index if pyg_graph.edge_index.size(0) > 0 else ([], [])
            try:
                dgl_graph = dgl.graph((src, dst), num_nodes=pyg_graph.x.shape[0])
              
                if hasattr(pyg_graph, 'x'):
                  
                    dgl_graph.ndata['node_attr'] = pyg_graph.x
                    dgl_graph.ndata['node_attr'] = dgl_graph.ndata['node_attr'].float() # float32
                if hasattr(pyg_graph, 'edge_attr'):
                    dgl_graph.edata['edge_attr'] = pyg_graph.edge_attr
            
               
                label = int(pyg_graph.y.item()) 
                attack_data.append((dgl_graph, label, filename))
            except:
                print('error:{}'.format(filename))
    
    return attack_data



def black_attack(model, device, graph, num_node=1, threshold = 0.1, noise_std = 0.1):
    model.eval()
    predictions = None
    size = graph.num_nodes()
    nxg = nx.Graph(graph.to_networkx())
    page = pagerank(nxg)
    between = betweenness(nxg)
    PAGERANK = sorted([(page[i], i) for i in range(size)], reverse=True)
    BETWEEN = sorted([(between[i], i) for i in range(size)], reverse=True)
    bar, Baseline_Degree, Baseline_Random = getThrehold(graph, size, threshold, num_node)
    
    for i, targets in enumerate([Baseline_Degree]):  
        data = deepcopy(graph)
        for target in targets:
            features = data.ndata['node_attr'][target]
            noise = torch.randn_like(features) * noise_std
            data.ndata['node_attr'][target] = features + noise
        with th.no_grad():
            data = data.to(device)
            logits = model(data)
            predictions = logits[:, 0].cpu().numpy()
            # print (predictions)
    return predictions[0], data




def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='Attack_argmax')
    parser.add_argument('-train_dir', required=True, help='Input train dataset directory')
    parser.add_argument('-test_dir', required=True, help='Input test dataset directory')
    parser.add_argument('-attack_dir', required=True, help='Input attack dataset directory')
    parser.add_argument('-record_dir', required=True, help='Record Directory') 
    parser.add_argument('-save_dir', required=True, help='Save Directory') 

    args = parser.parse_args()
    record_dir = args.record_dir
    save_dir = args.save_dir
    attack_dir = args.attack_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
   
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    max_queries = 1000
    
    
    os.makedirs(record_dir, exist_ok=True)
    results_path = os.path.join(record_dir, "attack_node_feats_final.csv")
    os.makedirs(save_dir, exist_ok=True)
    

   
    attack_loader = load_attack_data(attack_dir)
    
    
    model = train_GIN_model(train_dir, test_dir, device)
    model.eval()
    
    results = []
    ben_counts = 0
    one_counts = 0
    # def black_attack(model, graph, num_node=1, noise_std = 0.1):
    for graph, label, filename in tqdm(attack_loader, desc="Attacking"):
        
        # print (filename)
        if graph.number_of_nodes() == 1:
            one_counts += 1
            print ('nodes of one then skip')
            continue
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        graph, label = graph.to(device), torch.tensor(label, dtype=torch.long).to(device)
        predictions = model(graph).detach().cpu().numpy()
        predictions = predictions[:, 0]
        if predictions[0] < 0:
            ben_counts += 1
            print ('pred ben then skip')
            continue
        
        graph = graph.to("cpu")
        
        
        for i in range(max_queries):
            pred, adv_sample = black_attack(model, device, graph) 
            if pred < 0:
                results.append((filename, nodes, edges, i+1))
                save_adversarial_example(filename, adv_sample, save_dir)
                break
            
    with open(results_path, 'w') as file:
        file.write('filename, nodes, edges, queries\n')
        for i in results:
            file.write(','.join(map(str, i)) + '\n')
          
    print(f"Attack results saved to {results_path}")

    ori_num = len([f for f in os.listdir(attack_dir)])
    adv_num = len([f for f in os.listdir(save_dir)])
    print(f"ori_num is {ori_num}, adv_num is {adv_num}")
    print(f"ben num is {ben_counts}, one node_num is {one_counts}")



    
if __name__ == "__main__":
    main()
    # load_data()
