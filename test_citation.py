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

dataset = PygLinkPropPredDataset(name = "ogbl-citation", transform=T.ToSparseTensor())
device = gpu_setup(True, 0)
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_edge = dataset.get_edge_split()
torch.manual_seed(1)
idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
split_edge['eval_train'] = {
    'source_node': split_edge['train']['source_node'][idx],
    'target_node': split_edge['train']['target_node'][idx],
    'target_node_neg': split_edge['valid']['target_node_neg'],
}
print(split_edge['train']['source_node'].size())
print(split_edge['train']['source_node'])
print(split_edge['train']['target_node'].size())
print(split_edge['train']['target_node'])
print(split_edge['valid']['source_node'].size())
print(split_edge['valid']['source_node'])
print(split_edge['valid']['target_node'].size())
print(split_edge['valid']['target_node'])
print(split_edge['valid']['target_node_neg'].size())
print(split_edge['valid']['target_node_neg'])
print(split_edge['test']['source_node'].size())
print(split_edge['test']['source_node'])
print(split_edge['test']['target_node'].size())
print(split_edge['test']['target_node'])
print(split_edge['test']['target_node_neg'].size())
print(split_edge['test']['target_node_neg'])
'''
f = open("citation/train_pos_edge.txt","w")
ret = split_edge['train']['source_node']
for x in ret:
    f.write(str(x.item())+"\n")
ret = split_edge['train']['target_node']
for x in ret:
    f.write(str(x.item())+"\n")
f.close()
train_neg_edge = torch.randint(0, data.num_nodes, split_edge['train']['source_node'].size(), dtype=torch.long)
f = open("citation/train_neg_edge.txt","w")
for x in train_neg_edge:
    f.write(str(x.item())+"\n")
train_neg_edge = torch.randint(0, data.num_nodes, split_edge['train']['source_node'].size(), dtype=torch.long)
for x in train_neg_edge:
    f.write(str(x.item())+"\n")
f.close()
'''
f = open("citation/valid_edge.txt","w")
ret = split_edge['valid']['source_node']
for x in ret:
    f.write(str(x.item())+"\n")
ret = split_edge['valid']['target_node']
for x in ret:
    f.write(str(x.item())+"\n")
f.close()
ret = split_edge['valid']['target_node_neg']
for i in range(1000):
    tmp = ret[i]
    f = open("citation/valid_neg"+str(i)+".txt","w")
    for x in tmp:
        f.write(str(x.item())+"\n")
    f.close()
'''
ret = split_edge['valid']['target_node_neg'].reshape(-1)
for x in ret:
    f.write(str(x.item())+"\n")
f.close()
'''
f = open("citation/test_edge.txt","w")
ret = split_edge['test']['source_node']
for x in ret:
    f.write(str(x.item())+"\n")
ret = split_edge['test']['target_node']
for x in ret:
    f.write(str(x.item())+"\n")
f.close()
ret = split_edge['test']['target_node_neg']
for i in range(1000):
    tmp = ret[i]
    f = open("citation/test_neg"+str(i)+".txt","w")
    for x in tmp:
        f.write(str(x.item())+"\n")
    f.close()
'''
ret = split_edge['test']['target_node_neg'].reshape(-1)
for x in ret:
    f.write(str(x.item())+"\n")
f.close()
'''


'''
train_pos_edge = split_edge['train']['edge']
train_neg_edge = torch.randint(0, data.num_nodes, train_pos_edge.size(), dtype=torch.long)
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
