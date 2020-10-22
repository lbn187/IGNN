import torch
import argparse
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from logger import Logger
from mymodel import MyModel
from random_tree import Union, Tree, RandomGraph

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout
        self.num_layers=num_layers
        self.edge_layers = torch.nn.ModuleList()
        self.edge_layers.append(torch.nn.Linear(1, 1))
        self.concat_layers = torch.nn.ModuleList()
        self.concat_layers.append(torch.nn.Linear(2,1))
    def forward(self, x_i, x_j, edge_info):
        for layer in self.edge_layers[:-1]:
            edge_info = layer(edge_info)
            edge_info = F.relu(edge_info)
            edge_info = F.dropout(edge_info, p=self.dropout, training=self.training)
        edge_info = self.edge_layers[-1](edge_info)
        y = x_i * x_j
        for layer in self.layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.layers[-1](y)
        y = torch.cat([y, edge_info], dim = 1)
        for layer in self.concat_layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.concat_layers[-1](y)
        return torch.sigmoid(y)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.edge_layers:
            layer.reset_parameters()
        for layer in self.concat_layers:
            layer.reset_parameters()

def train(model, predictor, x, adj_t, train_pos_edge, train_pos_edge_info, train_neg_edge, train_neg_edge_info, optimizer, batch_size, device):
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col,row], dim = 0)
    model.train()
    predictor.train()
    total_loss = 0.0
    total_examples = 0
    perms = []
    cnt = 0
    for perm in DataLoader(range(train_pos_edge.size(0)), batch_size, shuffle=True):
        perms.append(perm)
    for perm in DataLoader(range(train_pos_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(x, adj_t)
        edge = train_pos_edge[perm].t()
        edge_info = train_pos_edge_info[perm].to(device)
        pos_out = predictor(h[edge[0]], h[edge[1]], edge_info)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        #edge = train_neg_edge[perms[cnt]].t()
        edge_info = train_neg_edge_info[perms[cnt]].to(device)
        cnt = cnt + 1
        #edge = train_neg_edge[perm].t()

        #edge_info = train_neg_edge_info[perm].to(device)
        edge = negative_sampling(edge_index, num_nodes=x.size(0), num_neg_samples = perm.size(0), method='dense')
        #edge = train_neg_edge[perm].t()
        neg_out = predictor(h[edge[0]], h[edge[1]], edge_info)
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, x, adj_t, valid_pos_edge, valid_pos_edge_info, valid_neg_edge, valid_neg_edge_info, test_pos_edge, test_pos_edge_info, test_neg_edge, test_neg_edge_info, evaluator, batch_size, device):
    model.eval()
    predictor.eval()
    h = model(x, adj_t)
    total_valid_loss = 0.0
    total_valid_examples = 0
    total_test_loss = 0.0
    total_test_examples = 0
    '''
    pos_train_preds = []
    for perm in DataLoader(range(train_pos_edge.size(0)), batch_size):
        edge = train_pos_edge[perm].t()
        edge_info = train_pos_edge_info[perm].to(device)
        pos_train_preds += [predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    print(pos_train_pred.size())
    print(pos_train_pred)
    pos_train_preds = []
    
    neg_train_preds = []
    for perm in DataLoader(range(train_neg_edge.size(0)), batch_size):
        edge = train_neg_edge[perm].t()
        edge_info = train_neg_edge_info[perm].to(device)
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]], edge_info).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)
    neg_train_preds = []
    '''
    pos_valid_preds = []
    for perm in DataLoader(range(valid_pos_edge.size(0)), batch_size):
        edge = valid_pos_edge[perm].t()
        edge_info = valid_pos_edge_info[perm].to(device)
        pos_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
        pos_valid_preds += [pos_out.squeeze().cpu()]
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        num_examples = pos_out.size(0)
        total_valid_loss += pos_loss.item() * num_examples
        total_valid_examples += num_examples
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    pos_valid_preds = []
    neg_valid_preds = []
    for perm in DataLoader(range(valid_neg_edge.size(0)), batch_size):
        edge = valid_neg_edge[perm].t()
        edge_info = valid_neg_edge_info[perm].to(device)
        neg_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
        neg_valid_preds += [neg_out.squeeze().cpu()]
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        total_valid_loss += neg_loss.item() * neg_out.size(0)
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    neg_valid_preds = []
    pos_test_preds = []
    for perm in DataLoader(range(test_pos_edge.size(0)), batch_size):
        edge = test_pos_edge[perm].t()
        edge_info = test_pos_edge_info[perm].to(device)
        pos_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
        pos_test_preds += [pos_out.squeeze().cpu()]
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        num_examples = pos_out.size(0)
        total_test_loss += pos_loss.item() * num_examples
        total_test_examples += num_examples
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_preds = []
    for perm in DataLoader(range(test_neg_edge.size(0)), batch_size):
        edge = test_neg_edge[perm].t()
        edge_info = test_neg_edge_info[perm].to(device)
        neg_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
        neg_test_preds += [neg_out.squeeze().cpu()]
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        total_test_loss += neg_loss.item() * neg_out.size(0)
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    valid_loss = total_valid_loss / total_valid_examples
    test_loss = total_test_loss / total_test_examples
    return valid_loss, test_loss, results
def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=list, default=[2])
    parser.add_argument('--in_newx', type=int, default=500)
    parser.add_argument('--hidden_channels', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=70000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--use_save', type=bool, default=False)
    parser.add_argument('--starteval_epoch', type=int, default=50)
    parser.add_argument('--use_res', type=bool, default=False)
    parser.add_argument('--num_trees', type=int, default=1)
    args = parser.parse_args()
    print(args)
    device = gpu_setup(True, args.device)
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0].to(device)
    adj_t = data.adj_t.to(device)
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)
    split_edge = dataset.get_edge_split()
    emb = torch.nn.Embedding(data.num_nodes, args.in_newx).to(device)
    train_pos_edge = split_edge['train']['edge'].to(device)
    train_neg_edge = train_pos_edge
    valid_pos_edge = split_edge['valid']['edge'].to(device)
    valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
    test_pos_edge = split_edge['test']['edge'].to(device)
    test_neg_edge = split_edge['test']['edge_neg'].to(device)
    print(valid_pos_edge.size())
    print(valid_neg_edge.size())
    print(test_pos_edge.size())
    print(test_neg_edge.size())
    train_pos_edge_info = torch.FloatTensor(train_pos_edge.size(0), args.num_trees).to(device)
    train_neg_edge_info = torch.FloatTensor(train_neg_edge.size(0), args.num_trees).to(device)
    valid_pos_edge_info = torch.FloatTensor(valid_pos_edge.size(0), args.num_trees).to(device)
    valid_neg_edge_info = torch.FloatTensor(valid_neg_edge.size(0), args.num_trees).to(device)
    test_pos_edge_info = torch.FloatTensor(test_pos_edge.size(0), args.num_trees).to(device)
    test_neg_edge_info = torch.FloatTensor(test_neg_edge.size(0), args.num_trees).to(device)
    f = open("ddi/train_pos_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    train_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(train_pos_edge.size(0), args.num_trees)) 
    f.close()
    f = open("ddi/train_neg_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    train_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(train_neg_edge.size(0), args.num_trees))
    f.close()
    f = open("ddi/valid_pos_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    valid_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(valid_pos_edge.size(0), args.num_trees))
    f.close()
    f = open("ddi/valid_neg_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    valid_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(valid_neg_edge.size(0), args.num_trees))
    f.close()
    f = open("ddi/test_pos_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    test_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(test_pos_edge.size(0), args.num_trees))
    f.close()
    f = open("ddi/test_neg_anchordis.txt","r")
    lines = f.readlines()
    ret = [float(x)*0.00001 for x in lines]
    test_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(test_neg_edge.size(0), args.num_trees))
    f.close()
    print(train_pos_edge_info.mean())
    print(train_neg_edge_info.mean())
    print(valid_pos_edge_info.mean())
    print(valid_neg_edge_info.mean())
    print(test_pos_edge_info.mean())
    print(test_neg_edge_info.mean())
    
    model = MyModel(args.in_newx, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout, device).to(device)
    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }
    max_test_list = []
    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        max_test_hits = 0.0
        if args.use_save:
            model.load_state_dict(torch.load('./ddi_model'))
            precitor.load_state_dict(torch.load('./ddi_predictor'))
            emb.load_state_dict(torch.load('./ddi_x'))
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + list(emb.parameters()), lr=args.lr)
        #optimizer = torch.optim.Adam(list(predictor.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, train_pos_edge, train_pos_edge_info, train_neg_edge, train_neg_edge_info, optimizer, args.batch_size, device)
            if epoch > args.starteval_epoch:
                valid_loss, test_loss, results = test(model, predictor, emb.weight, adj_t, valid_pos_edge, valid_pos_edge_info, valid_neg_edge, valid_neg_edge_info, test_pos_edge, test_pos_edge_info, test_neg_edge, test_neg_edge_info, evaluator, args.batch_size, device)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    train_hits, valid_hits, test_hits = result
                    if key == "Hits@20":
                        print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'ValidLoss: {valid_loss:.4f}, '
                          f'TestLoss: {test_loss:.4f}, '
                          f'Train: {100 * train_hits:.2f}%, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%, ')
                    if test_hits > max_test_hits and key == "Hits@20":
                        max_test_hits = test_hits
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
        max_test_list.append(max_test_hits)
    print("MAX LIST")
    print(max_test_list)
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

if __name__ == "__main__":
    main()
