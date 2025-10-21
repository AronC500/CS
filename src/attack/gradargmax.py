# argmax_attack https://github.com/xingchenwan/grabnel

import sys
import os

import dgl
import torch
from torch.utils.data import DataLoader, default_collate #
import torch.optim as optim #
from tqdm import tqdm
from copy import deepcopy
from grabnel.attack.base_attack import BaseAttack
from grabnel.attack.utils import number_of_correct_predictions, setseed, classification_loss #
from grabnel.models.utils import get_model_class #
import pandas as pd
import numpy as np
from itertools import product
from functools import lru_cache
import argparse
from dgl.data import DGLDataset

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

   
    feature_dim = dataset[0][0].ndata['node_attr'].shape[1]
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


def save_adversarial_example(filename, adv_graph, save_dir):
    adv_filename = "_".join(["adv"] + filename.split("_")[1:])
    adv_filepath = os.path.join(save_dir, adv_filename)  
    torch.save(adv_graph, adv_filepath)


class GradArgMax(BaseAttack):

    def __init__(self, classifier, loss_fn, mode='flip', **kwargs):
        super().__init__(classifier, loss_fn)  
        self.mode = mode

   
    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int, device, verbose=True):
        """Attack graph by flipping edge with maximum gradient (in absolute value)."""
       
        # graph = dgl.transform.remove_self_loop(graph)
        graph = dgl.remove_self_loop(graph)
        
        # save original graph and create a fully connected graph with binary edge weights which represent membership.
        unperturbed_graph = deepcopy(graph) #
        m = unperturbed_graph.number_of_edges() # 
        graph, edge_weights = self.prepare_input(graph) # 

        # unperturbed_graph = graph
        
        # fast access to edge_ids
        self.graph = graph
        self.edge_ids.cache_clear() # 

        # initialise variables
        flipped_edges = set()  
        losses = [] 
        correct_prediction = [] 
        queries = [] 
        edge_count_changes = [] 
        edge_operations = [] 
        modified_edges = []  
        progress_bar = tqdm(range(budget), disable=not verbose) 

        # edge id for self loops if they exist
        
        self_loops = self.has_self_loops(graph)
        if self_loops:
            self_loop_ids = graph.edge_ids(graph.nodes(), graph.nodes(), return_uv=True)

        is_binary = 2
        # flag = False
        # sequential attack
        for i in progress_bar:

            # forward and backward pass
            graph = graph.to(device)  # 
            edge_weights = edge_weights.to(device)  
            
            
            edge_weights = edge_weights.detach().requires_grad_(True)
            edge_weights.grad = None
            
            predictions = self.classifier(graph, edge_weights) 
            
           
            if predictions.shape[1] > 1:
                predictions = predictions[:, 0]
            label_prediction = number_of_correct_predictions(predictions, label, is_binary).detach().item()
            
            # onehot_label = label.squeeze()  # squeeze to ensure it is a scalar
            # if predictions.shape != onehot_label.shape:
            #     onehot_label = torch.nn.functional.one_hot(onehot_label.long(), num_classes=2).float().squeeze()
            # loss = self.loss_fn(predictions, onehot_label)
            loss = self.loss_fn(predictions, label)
            loss.backward() 

            # update loss/query information
            losses.append(round(loss.detach().item(), 6))
            queries.append(i)
            
            # stop early if attack is a success
            if label_prediction != label.item():
                correct_prediction.append(False)
                edge_count_changes.insert(0,"null")
                edge_operations.insert(0,"null")
                modified_edges.insert(0,"null")
                break
            else:
                correct_prediction.append(True)

            # So the argmax chooses the most negative gradient for edges that
            # exist in the original graph or the most positive gradient for non-existent edges.
            gradients = edge_weights.grad.detach()
            gradients[:m] = -1 * gradients[:m]

            # mask gradients for already flipped edges so they cant be selected
            for edge in flipped_edges:
                edge_ids = self.edge_ids(edge)
                gradients[edge_ids] = -np.inf

            # mask self loops
            if self_loops:
                gradients[self_loop_ids] = -np.inf

            # mask gradients based on mode
            if self.mode == 'flip':
                pass
            elif self.mode == 'add':
                gradients[:m] = -np.inf
            elif self.mode == 'remove':
                gradients[m:] = -np.inf
            else:
                raise NotImplementedError('Only supports flip, add, remove.')

            # select edge to be flipped based on which flip will increase loss the most
            try:
                edge_index = torch.argmax(gradients).item()
            except:
                # print(f"The graph number of nodes is {graph.number_of_nodes}, number of edges is {graph.number_of_edges}\n")
                edge_count_changes.insert(0,"null")
                edge_operations.insert(0,"null")
                modified_edges.insert(0,"null")
                break
            u = graph.edges()[0][edge_index].item()
            v = graph.edges()[1][edge_index].item()
            flipped_edge = frozenset((u, v))
            flipped_edges.add(flipped_edge)

            # update edge weights
            edge_weights = edge_weights.detach()
            edge_ids = self.edge_ids(flipped_edge)
            edge_weights[edge_ids] = 1 - edge_weights[edge_ids]
            
           
            if self.mode == 'flip':
                if (edge_weights[edge_ids] == 0).all():
                    edge_count_changes.append(-1)
                    edge_operations.append('del')
                    modified_edges.append((u, v))
                else:
                    edge_count_changes.append(1)
                    edge_operations.append('add')
                    modified_edges.append((u, v))
          
                
            # update tqdm progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4}', 'selected': (u, v)})

        # prepare output information
        df = pd.DataFrame({'queries': queries,
                            'losses': losses,
                           'edge_count_change': edge_count_changes,  
                           'edge_operation': edge_operations,  
                           'modified_edges': modified_edges,  
                            'correct_prediction': correct_prediction
                          })

        # construct adversarial example if the attack succeeds
        if not correct_prediction[-1]:
            adv_example = self.construct_perturbed_graph(unperturbed_graph, flipped_edges)
        else:
            adv_example = None

        # print if attack succeeded
        if verbose:
            if not correct_prediction[-1]:
                print('Attack success')
            else:
                print('Attack fail')

        return df, adv_example

    @staticmethod
    def prepare_input(graph):
        """Make graph fully connected but with zero weight edges where they don't exist"""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        to_add_u = []
        to_add_v = []
        for u, v in product(range(n), range(n)):
            if u != v and not graph.has_edges_between(u, v):
                to_add_u.append(u)
                to_add_v.append(v)
        edge_weights = torch.hstack((torch.ones(m), torch.zeros(len(to_add_u))))
        edge_weights.requires_grad = True
        graph.add_edges(to_add_u, to_add_v)
        return graph, edge_weights

 
    def construct_perturbed_graph(self, graph, flipped_edges):
        """Takes the unperturbed graph and list of flipped edges and applys the perturbation."""
        for edge in flipped_edges:
            u, v = edge
            if graph.has_edges_between(u, v):
                _, _, edge_ids_uv = graph.edge_ids([u], [v], return_uv=True)
                _, _, edge_ids_vu = graph.edge_ids([v], [u], return_uv=True)
                del_edges = torch.hstack((edge_ids_uv, edge_ids_vu))
                graph.remove_edges(list(del_edges.cpu().numpy()))
            else:
                graph.add_edges([u, v], [v, u])
        return graph

    @lru_cache(maxsize=None)
    def edge_ids(self, edge: frozenset):
        """Edge ids for edges u ~ v."""
        u, v = edge
        _, _, edge_ids_uv = self.graph.edge_ids([u], [v], return_uv=True)
        _, _, edge_ids_vu = self.graph.edge_ids([v], [u], return_uv=True)
        return torch.hstack((edge_ids_uv, edge_ids_vu))

    @staticmethod
    def has_self_loops(graph):
        """Determine if the graph contains self loops"""
        u = graph.nodes()
        return graph.has_edges_between(u, u).all().item()
    
    
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
    
 
    
    budget = 1000 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(record_dir, exist_ok=True)
    results_path = os.path.join(record_dir, "attack_argmax.csv") 
    os.makedirs(save_dir, exist_ok=True)
    
   
    attack_loader = load_attack_data(attack_dir)
    
  
    model = train_GIN_model(train_dir, test_dir, device)
    model.eval()

   
    num_classes = 2
    loss_fn = classification_loss(num_classes)
    attack = GradArgMax(classifier=model, loss_fn=loss_fn, mode="flip")

   
    results = []
    file_smi = []
    for graph, label, filename in tqdm(attack_loader, desc="Attacking"):
        graph, label = graph.to(device), torch.tensor(label, dtype=torch.long).to(device)
        attack_result, adv_graph = attack.attack(
            graph=graph,
            label=label,
            budget=budget,
            max_queries=1,
            device=device,
            verbose=False,
        )

      
        if adv_graph and attack_result.shape[0] != 1:
            save_adversarial_example(filename, adv_graph, save_dir)
            results.append(attack_result)  
            file_smi.append((filename, graph.number_of_nodes(), graph.number_of_edges())) 
        else:
            pass
            

   
    # all_results = pd.concat(results)
    with open(results_path, 'w') as file:
        file.write('filename, ori_graph_number_of_nodes, ori_graph_number_of_edges\n')
        file.write('queries, losses, edge_change, edge_operation, modified_edge, correct_prediction\n\n')
        for i, attack_result in enumerate(results):
            file.write(','.join(map(str, file_smi[i])) + '\n')
            attack_result.to_csv(file, index=False, header=file.tell() == 0)
            file.write('\n')
    # all_results.to_csv(results_path, index=False)
    print(f"Attack results saved to {results_path}")

    ori_num = len([f for f in os.listdir(attack_dir)])
    adv_num = len([f for f in os.listdir(save_dir)])
    print(f"ori_num is {ori_num}, adv_num is {adv_num}")
    
    
if __name__ == "__main__":
    main()
