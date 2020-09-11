import argparse
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
from randomtree import Union, Tree, RandomGraph

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, weights, add_self_weight, use_res, dropout, device):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        #self.layer_norm = torch.nn.LayerNorm(hidden_channels, elementwise_affine=True)
        self.num_layers = num_layers
        for i in range(len(num_layers)):
            num_layer = num_layers[i]
            convs = torch.nn.ModuleList()
            convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layer - 1):
                convs.append(SAGEConv(hidden_channels, hidden_channels))
            convs = convs.to(device)
            self.convs.append(convs)
        self.dropout = dropout
        self.add_self_weight = add_self_weight
        self.use_res = use_res
        self.weights = weights
    def reset_parameters(self):
        for convlist in self.convs:
            for conv in convlist:
                conv.reset_parameters()

    def forward(self, x, adj_t):
        x_all = []
        for convlist in self.convs:
            x1 = x
            x1 = convlist[0](x1, adj_t)
            for conv in convlist[1:]:
                x2 = x1
                x1 = F.relu(x1)
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
                if self.use_res:
                    x1 = conv(x1, adj_t) + x2
                else:
                    x1 = conv(x1, adj_t)
            x_all.append(x1)
        x_final = x_all[0] * self.weights[0]
        for i in range(1, len(x_all)):
            x_final += x_all[i] * self.weights[i]
        if self.add_self_weight > 1e-15:
            x_final += x * self.add_self_weight
        return x_final
    
    def inference(self, x_all, subgraph_loader, device):
        xlist = []
        for convlist in self.convs:
            xx = x_all
            for i, conv in enumerate(convlist): 
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, _, size = adj.to(device)
                    x = xx[n_id].to(device)
                    x_target = x[:size[1]]
                    if i == 0 :
                        x = conv((x, x[:size[1]]), edge_index)
                    else:
                        #x1 = self.layer_norm(x)
                        x = F.relu(x)
                        x = conv((x, x[:size[1]]), edge_index)
                        if self.use_res:
                            x += x_target
                    #if i == len(convlist) - 1:
                    #    x = F.relu(self.layer_norm(x))
                    xs.append(x.cpu())
                xx = torch.cat(xs, dim=0)
            xlist.append(xx)
        x_final = xlist[0] * self.weights[0]
        for i in range(1, len(xlist)):
            x_final += xlist[i] * self.weights[i]
        if self.add_self_weight > 1e-15:
            x_final += x_all.to('cpu') * self.add_self_weight
        return x_final

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, new_x, loader, optimizer, batch_size, device):
    model.to(device)
    model.train()
    predictor.train()
    total_loss = total_examples = 0
    for data, node_idx in loader:
        data = data.to(device)
        optimizer.zero_grad()
        concat_x = torch.cat([data.x.to(device), new_x[node_idx].to(device)], dim=1).to(device)
        h = model(concat_x, data.edge_index)
        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # Just do some trivial random sampling.
        #src_neg = src
        src_neg = torch.randint(0, data.x.size(0), src.size(), dtype=torch.long, device=device)
        dst_neg = torch.randint(0, data.x.size(0), src.size(), dtype=torch.long, device=device)
        neg_out = predictor(h[src_neg],h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def test(model, data, predictor, new_x, pos_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge, evaluator, subgraph_loader, batch_size, device):
    predictor.eval()
    model.eval()
    #data = data.to('cpu')
    #model = model.to('cpu')
    #data = data.to(device)
    #model = model.to(device)
    #h = model(data.x, data.edge_index)
    concat_x = torch.cat([data.x.to(device), new_x.to(device)], dim=1).to(device)
    h = model.inference(concat_x, subgraph_loader, device)
    pos_train_edge = pos_train_edge[:neg_valid_edge.size(0)]
    total_valid_loss = 0.0
    total_valid_examples = 0
    total_test_loss = 0.0
    total_test_examples = 0
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos0 = h[edge[0]].to(device)
        pos1 = h[edge[1]].to(device)
        pos_train_preds += [predictor(pos0, pos1).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos0 = h[edge[0]].to(device)
        pos1 = h[edge[1]].to(device)
        pos_out = predictor(pos0, pos1)
        pos_valid_preds += [pos_out.squeeze().cpu()]
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        num_examples = pos0.size(0)
        total_valid_loss += pos_loss.item() * num_examples
        total_valid_examples += num_examples
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        pos0 = h[edge[0]].to(device)
        pos1 = h[edge[1]].to(device)
        neg_out = predictor(pos0, pos1)
        neg_valid_preds += [neg_out.squeeze().cpu()]
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        num_examples = pos0.size(0)
        total_valid_loss += neg_loss.item() * num_examples

    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos0 = h[edge[0]].to(device)
        pos1 = h[edge[1]].to(device)
        pos_out = predictor(pos0, pos1)
        pos_test_preds += [pos_out.squeeze().cpu()]
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        num_examples = pos0.size(0)
        total_test_loss += pos_loss.item() * num_examples
        total_test_examples += num_examples
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        pos0 = h[edge[0]].to(device)
        pos1 = h[edge[1]].to(device)
        neg_out = predictor(pos0, pos1)
        neg_test_preds += [neg_out.squeeze().cpu()]
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        num_examples = pos0.size(0)
        total_test_loss += neg_loss.item() * num_examples
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
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

def main():
    parser = argparse.ArgumentParser(description='OGBL-PPA (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=list, default=[1,2,3])
    parser.add_argument('--in_newx', type=int, default=198)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--weights', type=list, default=[0.2, 0.35, 0.25])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--use_save', type=bool, default=False)
    parser.add_argument('--starteval_epoch', type=int, default=20)
    parser.add_argument('--add_self_weight', type=float, default=0.2)
    parser.add_argument('--use_res', type=bool, default=False)
    parser.add_argument('--num_trees', type=int, default=200) 
    args = parser.parse_args()
    print(args)
    device = gpu_setup(True, 0)
    dataset = PygLinkPropPredDataset(name='ogbl-ppa')
    data = dataset[0]
    new_x = torch.nn.Embedding(data.num_nodes, args.in_newx).to(device)
    data.x = data.x.to(torch.float)
    split_edge = dataset.get_edge_split()
    pos_train_edge = split_edge['train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']
    new_pos_train_edge = torch.tensor(pos_train_edge.size(0), args.num_trees + 2)
    new_pos_valid_edge = torch.tensor(pos_valid_edge.size(0), args.num_trees + 2)
    new_neg_valid_edge = torch.tensor(neg_valid_edge.size(0), args.num_trees + 2)
    new_pos_test_edge = torch.tensor(pos_test_edge.size(0), args.num_trees + 2)
    new_neg_test_edge = torch.tensor(neg_test_edge.size(0), args.num_trees + 2)
    randomgraph = RandomGraph(self.num_nodes)
    for i in range(pos_train_edge.size(0))
        x = pos_train_edge[i, 0]
        y = pos_train_edge[i, 1]
        randomgraph.add_edge(x, y)
        new_pos_train_edge[i, 0] = x
        new_pos_train_edge[i, 1] = y
    for i in range(pos_valid_edge.size(0))
        x = pos_valid_edge[i, 0]
        y = pos_valid_edge[i, 1]
        new_pos_valid_edge[i, 0] = x
        new_pos_valid_edge[i, 1] = y
    for i in range(neg_valid_edge.size(0))
        x = neg_valid_edge[i, 0]
        y = neg_valid_edge[i, 1]
        new_neg_valid_edge[i, 0] = x
        new_neg_valid_edge[i, 1] = y
    for i in range(pos_test_edge.size(0))
        x = pos_test_edge[i, 0]
        y = pos_test_edge[i, 1]
        new_pos_test_edge[i, 0] = x
        new_pos_test_edge[i, 1] = y
    for i in range(neg_test_edge.size(0))
        x = neg_test_edge[i, 0]
        y = neg_test_edge[i, 1]
        new_neg_test_edge[i, 0] = x
        new_neg_test_edge[i, 1] = y
    for nt in range(args.num_trees):
        tree = randomgraph.generate_random_tree()
        for i in range(pos_train_edge.size(0))
            x = pos_train_edge[i, 0]
            y = pos_train_edge[i, 1]
            new_pos_train_edge[i, nt + 2] = tree.dis(x, y)
        for i in range(pos_valid_edge.size(0))
            x = pos_valid_edge[i, 0]
            y = pos_valid_edge[i, 1]
            new_pos_valid_edge[i, nt + 2] = tree.dis(x, y)
        for i in range(neg_valid_edge.size(0))
            x = neg_valid_edge[i, 0]
            y = neg_valid_edge[i, 1]
            new_neg_valid_edge[i, nt + 2] = tree.dis(x, y)
        for i in range(pos_test_edge.size(0))
            x = pos_test_edge[i, 0]
            y = pos_test_edge[i, 1]
            new_pos_test_edge[i, nt + 2] = tree.dis(x, y)
        for i in range(neg_test_edge.size(0))
            x = neg_test_edge[i, 0]
            y = neg_test_edge[i, 1]
            new_neg_test_edge[i, nt + 2] = tree.dis(x, y)
     

    model = SAGE(data.num_features + args.in_newx, args.hidden_channels, args.num_layers, args.weights, args.add_self_weight, args.use_res, args.dropout, device).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout).to(device)
    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        torch.nn.init.xavier_uniform_(new_x.weight)
        if args.use_save:
            model.load_state_dict(torch.load('./save_model'))
            predictor.load_state_dict(torch.load('./save_predictor'))
            new_x.load_state_dict(torch.load('./save_x'))
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + list(new_x.parameters()),lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            sampler_data = data
            loader = GraphSAINTRandomWalkSampler(sampler_data, batch_size = args.batch_size, walk_length=3, num_steps=args.num_steps, sample_coverage=0, save_dir=dataset.processed_dir, num_workers=1)
            subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=4096, shuffle=False)
            loss = train(model, predictor, new_x.weight, loader, optimizer, 10000, device)
            print(loss)
            torch.save(model.state_dict(), './save_model')
            torch.save(predictor.state_dict(), './save_predictor')
            torch.save(new_x.state_dict(), './save_x')
            if epoch > args.starteval_epoch == 0:
                valid_loss, test_loss, results = test(model, data, predictor, new_x.weight, split_edge, evaluator, subgraph_loader, 10000, device)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'ValidLoss: {valid_loss:.4f}, '
                              f'TestLoss: {test_loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

if __name__ == "__main__":
    main()
