from math import log
import random
class Union(object):
    def __init__(self, n):
        super().__init__()
        self.n = n
        father = [i for i in range(n)]
    def find(self, x):
        if self.father[x] == x:
            return x
        else:
            self.father[x] = self.find(self.father[x])
    def is_connected(self, x, y):
        return self.find(x) == self.find(y)
    def union(self, x, y):
        self.father[self.find(x)] = self.find(y)

class Tree(object):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.vec = [[] for i in range(n)]
        self.deep = [0 for i in range(n)]
        self.logn = math.ceil(log(n, 2))
        self.fa = [[] for i in range(n)]
        for i in range(self.n):
            for j in range(self.logn):
                self.fa[i].append(-1)
    def add_edge(self, x, y):
        self.vec[x].append(y)
        self.vec[y].append(x)
    def dfs(self, x, father):
        for i in range(self.logn - 1):
            if self.fa[x][i] != -1
                self.fa[x][i + 1] = self.fa[self.fa[x][i]][i]
        for y in self.vec[x]:
            if y != father:
                self.deep[y] = self.deep[x] + 1
                self.fa[y][0] = x
                self.dfs(y, x)
    def lca(self, x, y):
        if d[x] < d[y]:
            x, y = y, x
        t = d[x] - d[y]
        for i in range(self.logn):
            if t >> i & 1:
                x = fa[x][i]
        for i in range(self.logn - 1, -1, -1):
            if fa[x][i] != fa[y][i]:
                x = fa[x][i]
                y = fa[y][i]
        if x == y:
            return x
        else:
            return fa[x][0]
    def dis(self, x, y):
        z = self.lca(x,y)
        return self.deep[x] + self.deep[y] - self.deep[z] * 2

class RandomGraph(object):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.m = 0
        self.edges = []
    def add_edge(self, x, y):
        self.edges.append([x, y])
    def generate_random_tree(self):
        index = [i for i in range(self.m)]
        bingchaji = Union(self.n)
        tree = Tree(self.n)
        for i in range(self.m):
            edge = self.edges[self.index[i]]
            x = edge[0]
            y = edge[1]
            if bingchaji.is_connected(x, y) == False:
                tree.add_edge(x, y)
                bingchaji.union(x, y)
        return tree
