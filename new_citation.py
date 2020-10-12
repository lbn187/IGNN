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
from gcnmodel import GCN

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
        self.edge_layers = torch.nn.ModuleList()
        self.edge_layers.append(torch.nn.Linear(1, 1))
        self.concat_layers = torch.nn.ModuleList()
        self.concat_layers.append(torch.nn.Linear(2, 1))
        self.dropout = dropout
        self.num_layers=num_layers
    def forward(self, x_i, x_j, edge_info):
        edge_info = self.edge_layers[0](edge_info)
        y = x_i * x_j
        for layer in self.layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.layers[-1](y)
        y = torch.cat([y, edge_info], dim=1)
        y = self.concat_layers[0](y)
        return torch.sigmoid(y)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.concat_layers:
            layer.reset_parameters()
        for layer in self.edge_layers:
            layer.reset_parameters()

def train(model, predictor, data, split_edge, pos_edge_info, neg_edge_info, optimizer, batch_size):
    model.train()
    predictor.train()
    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)
    total_loss = total_examples = 0
    cnt = 0
    perms = []
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        perms.append(perm)
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(data.x, data.adj_t)
        src, dst = source_edge[perm], target_edge[perm]
        edge_info = pos_edge_info[perm]
        pos_out = predictor(h[src], h[dst], edge_info)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        dst_neg = torch.randint(0, data.num_nodes, src.size(), dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        edge_info = neg_edge_info[perms[cnt]]
        cnt = cnt + 1
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, data, split_edge, valid_pos_edge_info, valid_neg_edge_info, test_pos_edge_info, test_neg_edge_info, evaluator, batch_size):
    '''
    predictor.eval()
    h = model(data.x, data.adj_t)
    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)
        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)
        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()
    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')
    return train_mrr, valid_mrr, test_mrr
    '''
    predictor.eval()
    h = model(data.x, data.adj_t)
    source = split_edge['valid']['source_node'].to(h.device)
    target = split_edge['valid']['target_node'].to(h.device)
    target_neg = split_edge['valid']['target_node_neg'].to(h.device)
    pos_preds = []
    for perm in DataLoader(range(source.size(0)), batch_size):
        src, dst = source[perm], target[perm]
        edge_info = valid_pos_edge_info[perm]
        pos_preds += [predictor(h[src], h[dst], edge_info).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)
    neg_preds = []
    source = source.view(-1, 1).repeat(1, 1000).view(-1)
    target_neg = target_neg.view(-1)
    for perm in DataLoader(range(source.size(0)), batch_size):
        src, dst_neg = source[perm], target_neg[perm]
        edge_info = valid_neg_edge_info[perm]
        neg_preds += [predictor(h[src], h[dst_neg], edge_info).squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()
    source = split_edge['test']['source_node'].to(h.device)
    target = split_edge['test']['target_node'].to(h.device)
    target_neg = split_edge['test']['target_node_neg'].to(h.device)
    pos_preds = []
    for perm in DataLoader(range(source.size(0)), batch_size):
        src, dst = source[perm], target[perm]
        edge_info = test_pos_edge_info[perm]
        pos_preds += [predictor(h[src], h[dst], edge_info).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)
    neg_preds = []
    source = source.view(-1, 1).repeat(1, 1000).view(-1)
    target_neg = target_neg.view(-1)
    for perm in DataLoader(range(source.size(0)), batch_size):
        src, dst_neg = source[perm], target_neg[perm]
        edge_info = test_neg_edge_info[perm]
        neg_preds += [predictor(h[src], h[dst_neg], edge_info).squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)
    test_mrr = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()
    return valid_mrr, test_mrr

def main():
    parser = argparse.ArgumentParser(description='OGBL-CITATION(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=list, default=[3])
    #parser.add_argument('--in_newx', type=int, default=500)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--use_save', type=bool, default=False)
    parser.add_argument('--starteval_epoch', type=int, default=0)
    parser.add_argument('--use_res', type=bool, default=True)
    parser.add_argument('--num_trees', type=int, default=1)
    args = parser.parse_args()
    print(args)
    device = gpu_setup(True, args.device)
    dataset = PygLinkPropPredDataset(name='ogbl-citation', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    split_edge = dataset.get_edge_split()
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }
    f = open("/blob2/v-bonli/citation/IGNN/citation/train_pos_edgeinfo.txt","r")
    lines = f.readlines()
    ret = [float(x) * 0.00005 for x in lines]
    train_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    f.close()
    f = open("/blob2/v-bonli/citation/IGNN/citation/train_neg_edgeinfo.txt","r")
    lines = f.readlines()
    ret = [float(x) * 0.00005 for x in lines]
    train_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    f.close()
    f = open("/blob2/v-bonli/citation/IGNN/citation/valid_pos_edgeinfo.txt","r")
    lines = f.readlines()
    ret = [float(x) * 0.00005 for x in lines]
    valid_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    f.close()
    f = open("/blob2/v-bonli/citation/IGNN/citation/test_pos_edgeinfo.txt","r")
    lines = f.readlines()
    ret = [float(x) * 0.00005 for x in lines]
    test_pos_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    f.close()
    ret = []
    for i in range(1000):
        f = open("/blob2/v-bonli/citation/IGNN/citation/valid_neg_edgeinfo"+str(i)+".txt","r")
        lines = f.readlines()
        ret.append([float(x) * 0.00005 for x in lines])
        f.close()
    valid_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    ret = []
    for i in range(1000):
        f = open("/blob2/v-bonli/citation/IGNN/citation/test_neg_edgeinfo"+str(i)+".txt","r")
        lines = f.readlines()
        ret.append([float(x) * 0.00005 for x in lines])
        f.close()
    test_neg_edge_info = torch.FloatTensor(np.array(ret).reshape(-1, 1)).to(device)
    
    model = GCN(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout, device).to(device)
    evaluator = Evaluator(name='ogbl-citation')
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
        #optimizer = torch.optim.Adam(list(predictor.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, train_pos_edge_info, train_neg_edge_info, optimizer, args.batch_size)
            if epoch > args.starteval_epoch:
                result = test(model, predictor, data, split_edge, valid_pos_edge_info, valid_neg_edge_info, test_pos_edge_info, test_neg_edge_info, evaluator, args.batch_size)
                logger.add_result(run, result)
                train_mrr, valid_mrr, test_mrr = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_mrr:.4f}, '
                      f'Valid: {valid_mrr:.4f}, '
                      f'Test: {test_mrr:.4f}, ')
        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()
