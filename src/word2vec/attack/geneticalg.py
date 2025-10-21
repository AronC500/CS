"""Genetic algorithm attack."""
from copy import deepcopy
import os
import sys
import dgl
import numpy as np
import pandas as pd
import scipy
import torch

_current_dir = os.path.abspath(os.path.dirname(__file__)) 
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, '..')) 
sys.path.append(PROJECT_ROOT) 


from tqdm import tqdm
from copy import deepcopy

from grabnel.attack.base_attack import BaseAttack
from grabnel.attack.utils import classification_loss,correct_predictions, population_graphs, random_sample_flip, random_sample_rewire_swap, get_allowed_nodes_k_hop, extrapolate_breakeven, nettack_loss, number_of_correct_predictions
from dgl.data import DGLDataset #
from torch.utils.data import DataLoader, default_collate #
import torch.optim as optim #
from grabnel.attack.utils import number_of_correct_predictions, setseed, classification_loss #
from grabnel.models.utils import get_model_class #
import argparse #

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
                        dgl_graph.ndata['node_attr'] = dgl_graph.ndata['node_attr'].float()
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

    # 获取特征维度和类别数量
    feature_dim = dataset[0][0].ndata['node_attr'].shape[1] # node_attr
    labels = torch.tensor([x[1] for x in dataset])
    num_classes = torch.max(labels).item() + 1

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)


def collate(samples):
    # `samples` 是一个列表，其中包含了被 `Dataset.__getitem__` 方法返回的数据对
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels)  # 对于标签，可以使用default_collate
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


class Genetic(BaseAttack):

    def __init__(self, classifier, loss_fn, population_size: int = 100,
                 crossover_rate: float = 0.1, mutation_rate: float = 0.2,
                 target_class: int = None,
                 mode: str = 'flip'):
        """A genetic algorithm based attack.

        This class stores an unperturbed graph in dgl.DGLGraph format, but the perturbed samples are represented as set.
        Each sample is a set of tuples (u, v) where u < v which represents an undirected edge u ~ v. To realise this
        perturbation each of the edges in the set are flipped. The original graph will be referred to as `graph` and an
        element of the population `sample`.

        Args:
            classifier: see BaseAttack
            loss_fn: see BaseAttack
            population_size: The number of perturbed graph in the population at any one point.
            crossover_rate: `crossover_rate` x `population_size` of the samples will be crossed over in each step.
            mutation_rate: All samples are mutated, `mutation_rate` of the flipped edges will be mutated.
            :param mode: str: 'flip', 'add', 'remove' or 'rewire': allowed edit operations on the edges.
        """
        super().__init__(classifier, loss_fn)
        self.target_class = target_class
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        assert mode in ['flip', 'add', 'remove', 'rewire'], f'mode {mode} is not recognised!'
        self.mode = mode

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int, device, verbose=True,
               initial_population: list = None):
        """The attack proceeds by rounds of selection, crossover and mutation. The number of rounds is determined by
        the `population_size` and `max_queries`. The population is a list of sets. Each set represents a perturbation.
        The set is of edge pairs (u, v) where u < v.

        initial_population: list: if specified, use this list of samples as initial population. Otherwise we randomly
            sample from the graphs
        """
        # print ('start attacking')
        total_round = 0
        graph = dgl.remove_self_loop(graph)
        adv_example = None
        is_edge_weighted = 'weight' in graph.edata.keys()
        if initial_population is not None:
            self.population += initial_population
            if len(self.population) < self.population_size:
                if self.mode == 'rewire':
                    self.population += [random_sample_rewire_swap(graph, budget, rewire_only=not is_edge_weighted)
                                        for _ in
                                        range(self.population_size - len(self.population))]
                else:
                    self.population += [random_sample_flip(graph, budget) for _ in
                                        range(self.population_size - len(self.population))]
        else:
            self.population = self.initial_population(graph, budget)
        rounds = max(1, np.round(max_queries / self.population_size).astype(int))
        # print (rounds)
        merged_dfs = None
        query_nums = None
        best_losses_so_far = []
        for round_no in range(rounds):
            fitness, predictions = self.fitness_of_population(graph, label, self.population, device, verbose=True)
            # ypred = torch.argmax(predictions, dim=1)
            # print(ypred)
            # print (label)
            # print('fitness:', fitness)
            fitness = np.nan_to_num(fitness, neginf=0., posinf=0.)
            # print('fitness:', fitness)
            print(f'Round{round_no}/{rounds}: {np.max(fitness)}')
            self.population = self.select_fittest(self.population, fitness)
            self.population = self.crossover_population(self.population, budget)
            self.population = self.mutate_population(graph, self.population)
            new_df = self.construct_dataframe(fitness, predictions, label.squeeze(), (round_no + 1) * self.population_size)
            if merged_dfs is None: merged_dfs = new_df
            else: merged_dfs = pd.concat([merged_dfs, new_df])
            # added by xingchen: terminate the run whenever the attack succeeds.
            labels = torch.repeat_interleave(label, len(predictions))
            if (self.target_class is None and np.sum(correct_predictions(predictions.cpu().numpy(), labels.cpu().numpy())) < len(predictions)):
                
                total_round = round_no
                correct_nums = np.sum(correct_predictions(predictions.cpu().numpy(), labels.cpu().numpy()))
                query_nums = total_round * self.population_size + correct_nums
                print('Attack succeeded! Total queries:',query_nums)
                # ypred = torch.argmax(predictions, dim=1)
                # print(ypred)
                # print (labels)
                if self.target_class is None:
                    # comps = correct_predictions(predictions.numpy(), labels.numpy())
                    comps = correct_predictions(predictions.cpu().numpy(), labels.cpu().numpy())
                    for i, comp in enumerate(comps):
                        if not comp:
                            adv_example = population_graphs(graph, [self.population[i]], mode=self.mode)
                            break
                else:
                    for i, pred in enumerate(predictions):
                        if np.argmax(pred.numpy()) == self.target_class:
                            adv_example = population_graphs(graph, [self.population[i]], mode=self.mode)
                            break
                break
            else:
                total_round = round_no + 1
                query_nums = total_round * self.population_size

            
            # ypred = torch.argmax(predictions, dim=1)
            # print(ypred)
            # print (labels)
            
            best_losses_so_far.append(np.max(merged_dfs.losses.values))
            if len(best_losses_so_far) > 200 / self.population_size and extrapolate_breakeven(best_losses_so_far) > 1e5 / self.population_size:
                print(f'Predicted breakeven point is {extrapolate_breakeven(best_losses_so_far)} and run terminated')
                break

        return merged_dfs, adv_example, query_nums

    def initial_population(self, graph: dgl.DGLGraph, budget: int) -> list:
        """Create an initial population using random flips to create perturbation."""
        is_edge_weighted = 'weight' in graph.edata.keys()
        if self.mode == 'rewire':
            population = [random_sample_rewire_swap(graph, budget, rewire_only=not is_edge_weighted) for _ in
                          range(self.population_size - len(self.population))]
        else:
            population = [random_sample_flip(graph, budget) for _ in range(self.population_size)]
        return population

    def fitness_of_population(self, graph: dgl.DGLGraph, label: torch.tensor, population: list, device, verbose=True) \
            -> (np.array, torch.tensor):
        """Evaluate the fitness of the population.

        Args:
            graph: The original unperturbed graph
            label: The label of the unperturbed graph
            population: A list of perturbed graphs

        Returns:
            fitness: A 1D numpy array where the ith element is the loss of element i in the population
            predictions: A torch array containing logits (1D if its a binary classification task, otherwise an (n x C)
            array where C is the number of classes.
        """
        perturbed_graphs = population_graphs(graph, population, self.mode)
        # perturbed_graphs = perturbed_graphs.to(device)
        with torch.no_grad():
            try:
                batched_graph = dgl.batch(perturbed_graphs)
                batched_graph = batched_graph.to(device)
                predictions = self.classifier(batched_graph)
            except RuntimeError:
                # this is possibly a dgl bug seemingly related to this https://github.com/dmlc/dgl/issues/2310
                # dgl.unbatch() should exactly inverses dgl.batch(), but you might get RuntimeError by doing something
                # like dgl.unbatch(dgl.batch([graphs])).
                
                batched_graph = dgl.batch(perturbed_graphs)
                batched_graph = batched_graph.to(device)
                predictions = self.classifier(batched_graph)
            # print (predictions.shape)
            if predictions.shape[1] > 1:
                predictions = predictions[:, 0]
                  
                
           
                # predictions = self.classifier(perturbed_graphs)
            # label_prediction = torch.argmax(predictions) 
            # ypred = torch.argmax(predictions, dim=1)
            labels = torch.repeat_interleave(label, len(perturbed_graphs))
            # print (ypred, labels)
            fitness = self.loss_fn(predictions, labels)
            # print ('fitness:', fitness)
            # print (predictions)
            # print (labels)
        fitness = fitness.detach().cpu().numpy()
        return fitness, predictions

    def select_fittest(self, population: list, fitness: np.array) -> list:
        """Takes half the fittest scores and then samples the other half using softmax weighting on the scores."""
        softmax_fitness = scipy.special.softmax(fitness)
        fittest_idx = np.argsort(-softmax_fitness)[:int(np.floor(self.population_size / 2))]
        population_size = int(self.population_size)
        half_population = int(np.ceil(population_size / 2))
        # print ('######### DEBUG #########')
        # print("fitness:", fitness, "shape:", fitness.shape, "dtype:", fitness.dtype)
        random_idx = np.random.choice(np.arange(population_size), half_population, replace=False, p=softmax_fitness)
        # random_idx = np.random.choice(np.arange(self.population_size), int(np.ceil(self.population_size / 2)),
        #                               replace=True, p=softmax_fitness)
        all_idx = np.concatenate([fittest_idx, random_idx])
        population = [population[idx] for idx in all_idx]
        
        
        # print ('######### DEBUG #########')
        # print (random_idx)
        # print (population)
        # print (len(population))
        
        return population

    def crossover_population(self, population: list, budget: int) -> list:
        """Each sample is crossed over by probability `self.crossover_rate`."""
        for i, sample in enumerate(population):
            if self.crossover_rate < np.random.rand():
                population[i] = self.crossover(sample, population, budget)
        return population

    def crossover(self, sample: set, population: list, budget: int) -> set:
        """Cross over of the `sample` and one other random sample from the `population`. The crossover is done by taking
        the union of all flips of the two samples and then sampling `budget` of them to create a new sample."""
        other_sample = np.random.choice(range(self.population_size))
        
        # print("other_sample:", other_sample, "population_length:", len(population))
        other_sample = population[other_sample]
        all_flips = list(set(sample).union(set(other_sample)))
        new_sample = np.random.choice(range(len(all_flips)), budget, replace=False)
        new_sample = set([all_flips[i] for i in new_sample])
        return new_sample

    def mutate_population(self, graph: dgl.DGLGraph, population: list) -> list:
        """Mutate all samples in the population."""
        for idx in range(self.population_size):
            population[idx] = self.mutate_sample(graph, population[idx])
        return population

    def mutate_sample(self, graph: dgl.DGLGraph, sample: set, ) -> set:
        """ Mutate the edges in the sample with at a rate of `self.mutation_rate`.

        Args:
            graph: The original unperturbed graph
            sample: The perturbed graph represented as a set of edge flips

        Returns:
            A new perturbed graph (in set format) which is a mutation of `sample`.
        """
        is_edge_weighted = 'weight' in graph.edata.keys()
        new_sample = set()

        # choose edges to mutate
        to_mutate = []
        for i, edge in enumerate(sample):
            if np.random.rand() < self.mutation_rate:
                to_mutate.append(edge)
            else:
                new_sample.add(edge)

        # mutate edges for new sample
        for edge in to_mutate:
            new_edge = self.mutate_rewire_triplet(graph, edge, rewire_only=not is_edge_weighted) \
                if self.mode == 'rewire' \
                else self.mutate_edge(graph, edge, )
            while new_edge in new_sample:
                new_edge = self.mutate_rewire_triplet(graph, edge, rewire_only=not is_edge_weighted) \
                    if self.mode == 'rewire' \
                    else self.mutate_edge(graph, edge,)
            new_sample.add(new_edge)

        return new_sample

    @staticmethod
    def mutate_edge(graph, edge, ):
        """Mutate a single edge.  The mutation chooses a random end point of the edge and then pairs it with a random
        node in the graph.
        """
        u, v = edge
        if np.random.rand() < 0.5:
            new_u = u
        else:
            new_u = v
        available_nodes = np.arange(graph.number_of_nodes())

        new_v = np.random.choice(available_nodes)
        while new_u == new_v:
            new_v = np.random.choice(available_nodes)

        return min(new_u, new_v), max(new_u, new_v)

    @staticmethod
    def mutate_rewire_triplet(graph, edge, rewire_only: bool = False, swap_only: bool = False):
        """Mutate triplet (u, v, w) used for rewiring operation (i.e. we either rewire u->v to u->w, or for the case
        when (u, w) is already an edge, swap u-v and u-w"""
        from copy import deepcopy
        if rewire_only and swap_only: raise ValueError(
            'Only either or neither of swap_only and rewire_only can be True!')
        # the index of the triplet to mutate
        patience = 100
        new_edge = deepcopy(edge)
        u, v, w = new_edge

        while patience >= 0:
            rewire_id = np.random.randint(0, len(edge))
            if rewire_id == 0:  # the candidate u is the neighbours of v with index number  < v
                new_node = np.random.choice(graph.out_edges(v)[1])
                if new_node in [u, v, w] or new_node > v:
                    patience -= 1
                    continue
                new_edge = (new_node, v, w)
                break
            elif rewire_id == 1:  # the candidate v is the neighbour of u with index number > u
                new_node = np.random.choice(graph.out_edges(u)[1])
                if new_node in [u, v, w] or new_node < u:
                    patience -= 1
                    continue
                new_edge = (u, new_node, w)
                break
            elif rewire_id == 2:
                if swap_only:
                    new_node = np.random.choice(graph.out_edges(u)[1])
                    if new_node in [u, v, w] or new_node < u:
                        patience -= 1
                        continue
                else:
                    new_node = np.random.randint(u, graph.number_of_nodes())
                    if new_node in [u, v, w]:
                        patience -= 1
                        continue
                    if rewire_only and new_node in graph.out_edges(u)[1]:
                        patience -= 1
                        continue
                new_edge = (u, v, new_node)
                break
        if patience <= 0:
            # print(f'Patience exhausted in trying to mutate {edge}!')
            return new_edge
        return new_edge

    @staticmethod
    def construct_dataframe(losses: np.array, predictions: torch.tensor, label: torch.tensor, queries: int) \
            -> pd.DataFrame:
        """Construct a pandas dataframe consistent with the base class. This dataframe is for all samples evaluated
        after exactly `queries` queries."""
        label_cpu = label.cpu().numpy()
        labels = np.tile(label_cpu, len(predictions))
        # labels = np.tile(label, len(predictions))
        df = pd.DataFrame({'losses': losses,
                           'correct_prediction': correct_predictions(predictions.cpu().numpy(), labels),
                           'queries': queries})
        return df


import signal


TIMEOUT = 300  


def timeout_handler(signum, frame):
    print("Attack timed out!")
    raise TimeoutError("Attack took too long to complete")

def main():
   
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
    
  
    
    budget = 10 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(record_dir, exist_ok=True)
    results_path = os.path.join(record_dir, 'attack_genetic.csv')
    os.makedirs(save_dir, exist_ok=True)
    
  
    attack_loader = load_attack_data(attack_dir)
    
   
    model = train_GIN_model(train_dir, test_dir, device)
    model.eval()

   
    num_classes = 2
    loss_fn = classification_loss(num_classes)
    
    attack = Genetic(classifier=model, loss_fn=loss_fn, population_size=100, mode="flip")
    
    

   
    signal.signal(signal.SIGALRM, timeout_handler)
    
   
    results = []
    total_rounds = []
    file_smi = []
    for graph, label, filename in tqdm(attack_loader, desc="Attacking"):
        # print (filename)
        if graph.number_of_nodes() == 1:
            print ('nodes of one then skip')
            continue
        
        # if filename in exist_advs_data:
        #     continue
       
        print ('file:',filename, ', nodes:', graph.number_of_nodes(), ', edges:', graph.number_of_edges())
        budget = int ((graph.number_of_nodes()+graph.number_of_edges())/2)
        
        graph, label = graph.to(device), torch.tensor(label, dtype=torch.long).to(device)
        
        predictions = model(graph).detach().cpu().numpy()
        predictions = predictions[:, 0]
        
        if predictions[0] < 0:
            print ('pred ben then skip')
            continue
        
            
        signal.alarm(TIMEOUT) 

        try:
            attack_result, adv_graph, total_queries = attack.attack(
                graph=graph,
                label=label,
                budget=budget,
                max_queries=1000,
                device=device,
                verbose=False,
            )
        except TimeoutError:
            print(f"Attack on {filename} timed out!")
            continue 

      
        signal.alarm(0) 
        
        
       
        if adv_graph and attack_result.shape[0] != 1:
            save_adversarial_example(filename, adv_graph, save_dir)
            results.append(attack_result)   
            file_smi.append((filename, graph.number_of_nodes(), graph.number_of_edges(), total_queries)) # 保存原始图信息
        else:
            pass
            

  
    # all_results = pd.concat(results)
    with open(results_path, 'w') as file:
        file.write('filename, ori_graph_number_of_nodes, ori_graph_number_of_edges, queries\n')
        # file.write('queries, losses, edge_change, edge_operation, modified_edge, correct_prediction\n\n')
        for i, attack_result in enumerate(results):
            file.write(','.join(map(str, file_smi[i])) + '\n')
            # attack_result.to_csv(file, index=False, header=file.tell() == 0)
            # file.write('\n')
    # all_results.to_csv(results_path, index=False)
    print(f"Attack results saved to {results_path}")

    ori_num = len([f for f in os.listdir(attack_dir)])
    adv_num = len([f for f in os.listdir(save_dir)])
    print(f"ori_num is {ori_num}, adv_num is {adv_num}")
    
    
    


    
if __name__ == "__main__":
    main()
