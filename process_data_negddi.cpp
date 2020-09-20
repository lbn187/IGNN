#include<bits/stdc++.h>

#define N 4444
#define M 1666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m, F[N];
int dis[N][N];
struct{
	int x,y;
}train_pos[M], train_pos2[M], train_neg[M], valid_pos[M], valid_neg[M], test_pos[M], test_neg[M];
int gf(int x){
	if(F[x]==x)return x;
	return F[x]=gf(F[x]);
}
struct Tree{
	vector<int>V[N];
	int F[N][15];
	int d[N];
	void ins(int x,int y){
		V[x].push_back(y);
		V[y].push_back(x);
	}
	void dfs(int x,int fa){
		for(int i=1;i<=12;i++)
			if(F[x][i-1]!=-1)
				F[x][i]=F[F[x][i-1]][i-1];
		for(int y:V[x])
			if(y!=fa){
				F[y][0]=x;
				d[y]=d[x]+1;
				dfs(y,x);
			}
	}
	int lca(int x,int y){
		if(d[x]<d[y])swap(x,y);
		int t=d[x]-d[y];
		for(int i=12;~i;i--)if(t>>i&1)x=F[x][i];
		for(int i=12;~i;i--)if(F[x][i]!=F[y][i])
			x=F[x][i],y=F[y][i];
		return x==y?x:F[x][0];
	}
	int dis(int x,int y){
		int z=lca(x,y);
		return d[x]+d[y]-2*d[z];
	}
}tree;
int main(){
	srand(time(0));
	n=4267;
	freopen("ddineg/all.txt","r",stdin);
	scanf("%d",&train_pos_m);
	for(int i=0;i<train_pos_m;i++)scanf("%d%d",&train_pos[i].x, &train_pos[i].y), train_pos2[i] = train_pos[i];
	scanf("%d",&valid_pos_m);
	for(int i=0;i<valid_pos_m;i++)scanf("%d%d",&valid_pos[i].x, &valid_pos[i].y);
	scanf("%d",&valid_neg_m);
	for(int i=0;i<valid_neg_m;i++)scanf("%d%d",&valid_neg[i].x, &valid_neg[i].y);
	scanf("%d",&test_pos_m);
	for(int i=0;i<test_pos_m;i++)scanf("%d%d",&test_pos[i].x, &test_pos[i].y);
	scanf("%d",&test_neg_m);
	for(int i=0;i<test_neg_m;i++)scanf("%d%d",&test_neg[i].x, &test_neg[i].y);
	fclose(stdin);
	/*
	for(int T=0;T<1000;T++){
		printf("%d\n",T);
		random_shuffle(train_pos2,train_pos2+train_pos_m);
		for(int i=0;i<n;i++)F[i]=i;
		for(int i=0;i<n;i++){
			tree.V[i].clear();
			tree.d[i]=0;
			for(int j=0;j<=12;j++)tree.F[i][j]=-1;
		}
		for(int i=0;i<train_pos_m;i++){
			int x=gf(train_pos2[i].x);
			int y=gf(train_pos2[i].y);
			if(x!=y){
				F[x]=y;
				tree.ins(train_pos2[i].x, train_pos2[i].y);
			}
		}
		tree.dfs(0, -1);
		for(int i=0;i<n;i++)
			for(int j=i+1;j<n;j++)
				dis[i][j] += tree.dis(i, j);
	}*/
	/*
	freopen("ddineg/dis_matrix.txt","w",stdout);
	for(int i=0;i<n;i++)for(int j=0;j<n;j++)printf("%d\n",i>j?dis[j][i]:dis[i][j]);
	fclose(stdout);
	*/
	freopen("ddineg/dis_matrix.txt","r",stdin);
	for(int i=0;i<n;i++)for(int j=0;j<n;j++)scanf("%d",&dis[i][j]);
	fclose(stdin);
	freopen("ddineg/train_pos_avg.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%d\n",dis[train_pos[i].x][train_pos[i].y]);
	fclose(stdout);
	freopen("ddineg/valid_pos_avg.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%d\n",dis[valid_pos[i].x][valid_pos[i].y]);
	fclose(stdout);
	freopen("ddineg/valid_neg_avg.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%d\n",dis[valid_neg[i].x][valid_neg[i].y]);
	fclose(stdout);
	freopen("ddineg/test_pos_avg.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%d\n",dis[test_pos[i].x][test_pos[i].y]);
	fclose(stdout);
	freopen("ddineg/test_neg_avg.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%d\n",dis[test_neg[i].x][test_neg[i].y]);
	fclose(stdout);
}
