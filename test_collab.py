import torch
import os
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset
from random_tree import Union, Tree, RandomGraph
import torch_geometric.transforms as T
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

dataset = PygLinkPropPredDataset(name = "ogbl-collab", transform=T.ToSparseTensor())
device = gpu_setup(True, 0)
data = dataset[0].to(device)
adj_t = data.adj_t.to(device)
row, col, _ = adj_t.coo()
edge_index = torch.stack([col, row], dim=0)
split_edge = dataset.get_edge_split()
train_pos_edge = split_edge['train']['edge']
train_neg_edge = torch.randint(0, data.num_nodes, train_pos_edge.size(), dtype=torch.long)
#train_neg_edge = negative_sampling(edge_index, num_nodes = data.num_nodes, num_neg_samples = train_pos_edge.size(0), method = 'dense').t()
'''
for i in range(train_neg_edge.size(0)):
    if train_neg_edge[i, 0] == train_neg_edge[i, 1]:
        train_neg_edge[i, 0] = train_neg_edge[i - 1, 0]
        train_neg_edge[i, 1] = train_neg_edge[i - 1, 1]
'''
valid_pos_edge = split_edge['valid']['edge']
valid_neg_edge = split_edge['valid']['edge_neg']
test_pos_edge = split_edge['test']['edge']
test_neg_edge = split_edge['test']['edge_neg']
f = open("collab/train_neg_edge.txt", "w")
for i in range(train_neg_edge.size(0)):
    x = train_neg_edge[i, 0].item()
    y = train_neg_edge[i, 1].item()
    if x == y:
        train_neg_edge[i] = train_neg_edge[i - 1]
negedge = train_neg_edge.reshape(-1)
print(negedge.size())
for x in negedge:
    f.write(str(x.item())+"\n")
f.close()
f = open("collab/all.txt", "w")
f.write(str(train_pos_edge.size(0))+"\n")
for i in range(train_pos_edge.size(0)):
    x = train_pos_edge[i, 0].item()
    y = train_pos_edge[i, 1].item()
    f.write(str(x)+" "+str(y)+"\n")
f.write(str(train_neg_edge.size(0))+"\n")
for i in range(train_neg_edge.size(0)):
    x = train_neg_edge[i, 0].item()
    y = train_neg_edge[i, 1].item()
    f.write(str(x) + " " + str(y) + "\n")
f.write(str(valid_pos_edge.size(0))+"\n")
for i in range(valid_pos_edge.size(0)):
    x = valid_pos_edge[i, 0].item()
    y = valid_pos_edge[i, 1].item()
    f.write(str(x) + " " + str(y) + "\n")
f.write(str(valid_neg_edge.size(0))+"\n")
for i in range(valid_neg_edge.size(0)):
    x = valid_neg_edge[i, 0].item()
    y = valid_neg_edge[i, 1].item()
    f.write(str(x) + " " + str(y) + "\n")
f.write(str(test_pos_edge.size(0))+"\n")
for i in range(test_pos_edge.size(0)):
    x = test_pos_edge[i, 0].item()
    y = test_pos_edge[i, 1].item()
    f.write(str(x) + " " + str(y) + "\n")
f.write(str(test_neg_edge.size(0))+"\n")
for i in range(test_neg_edge.size(0)):
    x = test_neg_edge[i, 0].item()
    y = test_neg_edge[i, 1].item()
    f.write(str(x) + " " + str(y) + "\n")
f.close()

'''
data = dataset[0]
print(data.x)
graph = RandomGraph(data.num_nodes, train_pos_edge)

for epoch in range(10):
    tree = graph.generate_random_tree()
    dis_pos = 0
    dis_neg = 0
    for i in range(test_pos_edge.size(0)):
        x = valid_pos_edge[i, 0]
        y = valid_pos_edge[i, 1]
        dis_pos += tree.dis(x, y)
    for i in range(test_neg_edge.size(0)):
        x = valid_neg_edge[i, 0]
        y = valid_neg_edge[i, 1]
        dis_neg += tree.dis(x,y)
    print(dis_pos)
    print(dis_neg)
'''
