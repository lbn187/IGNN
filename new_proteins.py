import argparse

import torch
import torch.nn.functional as F
import numpy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from mymodel import MyModel 

class NodePredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(NodePredictor, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels, out_channels))
        self.layers.to(device)
        self.node_layers = torch.nn.ModuleList()
        self.node_layers.append(torch.nn.Linear(1, out_channels))
        self.node_layers.append(torch.nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1))
        self.node_layers.to(device)
        self.concat_layers = torch.nn.ModuleList()
        for i in range(112):
            self.concat_layers.append(torch.nn.Linear(2, 1))
        self.concat_layers.to(device)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.node_layers:
            layer.reset_parameters()
        for layer in self.concat_layers:
            layer.reset_parameters()
    def forward(self, x, node_info):
        x = self.layers[0](x)
        y = self.node_layers[0](node_info)
        y = F.relu(y)
        y = self.node_layers[1](y)
        concat_v = x
        mylist=[]
        for i in range(112):
            concat_xy = torch.cat([x[:, i].reshape(-1, 1), y[:, i].reshape(-1, 1)], dim=1)
            concat_xy = self.concat_layers[i](concat_xy)
            mylist.append(concat_xy)
        return torch.cat(mylist, dim=1)

def train(model, predictor, data, node_info, train_idx, optimizer):
    model.train()
    predictor.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    out = predictor(out, node_info[train_idx])
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, predictor, data, node_info, split_idx, evaluator):
    model.eval()
    predictor.eval()
    y_pred = model(data.x, data.adj_t)
    y_pred = predictor(y_pred, node_info)
    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=list, default=[2,3])
    parser.add_argument('--hidden_channels', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_res', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    model = MyModel(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device)
    predictor = NodePredictor(args.hidden_channels, 112, device)
    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    f = open("proteins/node_distance.txt", "r")
    lines = f.readlines()
    ret = [float(x)-6.9 for x in lines]
    node_info = torch.FloatTensor(numpy.array(ret).reshape(data.num_nodes, 1)).to(device)
    f.close()
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, node_info, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, data, node_info, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
