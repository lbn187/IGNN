#include<bits/stdc++.h>
#define N 133333
#define M 40000000
using namespace std;
int n, m, nn, F[N], flag[N][115];
struct P{
	int x,y;
};
P edges[M];
double sum[N][115];
int R(){
	int x=rand()%32768;
	int y=rand()%32768;
	return x*32768+y;
}
int gf(int x){
	if(F[x]==x)return x;
	return F[x]=gf(F[x]);
}
struct Tree{
	vector<int>V[N];
	int F[N][18];
	int d[N];
	int sz[N][113];
	long long f[N][113];
	int n;
	void ins(int x,int y){
		V[x].push_back(y);
		V[y].push_back(x);
	}
	void dfs(int x,int fa){
		for(int i=1;i<=17;i++)
			if(F[x][i-1]!=-1)
				F[x][i]=F[F[x][i-1]][i-1];
		for(int i=0;i<=112;i++)if(flag[x][i])sz[x][i]=1;
		for(int y:V[x])
			if(y!=fa){
				F[y][0]=x;
				d[y]=d[x]+1;
				dfs(y,x);
				for(int i=0;i<=112;i++){
					sz[x][i]+=sz[y][i];
					f[x][i]+=f[y][i]+sz[y][i];
				}
			}
	}
	void dfs2(int x,int fa){
		for(int i=0;i<=112;i++)
			f[x][i]=f[fa][i]+sz[0][i]-2*sz[x][i];
		for(int y:V[x])
			if(y!=fa)dfs2(y,x);
	}
	int lca(int x,int y){
		if(d[x]<d[y])swap(x,y);
		int t=d[x]-d[y];
		for(int i=17;~i;i--)if(t>>i&1)x=F[x][i];
		for(int i=17;~i;i--)if(F[x][i]!=F[y][i])
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
	n=132534;
	nn=86619;
	freopen("proteins/edge.txt","r",stdin);
	m=39561252;
	for(int i=0;i<m;i++)scanf("%d%d",&edges[i].x,&edges[i].y);
	//for(int i=0;i<m;i++)
	//	scanf("%d%d",&edges[i].x, &edges[i].y);
	fclose(stdin);
	freopen("proteins/node.txt","r",stdin);
	for(int i=0;i<n;i++)
		for(int j=0;j<112;j++)
			scanf("%d",&flag[i][j]);
	for(int i=nn;i<n;i++)
		for(int j=0;j<112;j++)
			flag[i][j]=0;
	for(int i=0;i<n;i++)flag[i][112]=1;
	fclose(stdin);
	n++;
	for(int T=0;T<1000;T++){
		printf("%d\n",T);
		random_shuffle(edges, edges+m);
		for(int i=0;i<n;i++)F[i]=i;
		for(int i=0;i<n;i++){
			tree.V[i].clear();
			tree.d[i]=0;
			for(int j=0;j<=112;j++)tree.sz[i][j]=0, tree.f[i][j]=0;
			for(int j=0;j<=17;j++)tree.F[i][j]=-1;
		}
		int cnt=0;
		for(int i=0;i<m;i++){
			int x=gf(edges[i].x);
			int y=gf(edges[i].y);
			if(x!=y){
				F[x]=y;
				tree.ins(edges[i].x, edges[i].y);
				cnt++;
			}
		}
		for(int i=0;i<n-1;i++){
			int x=gf(i);
			int y=gf(n-1);
			if(x!=y){
				F[x]=y;
				tree.ins(i, n-1);
				cnt++;
			}
		}
		tree.n = n;
		tree.dfs(0, -1);
		for(int x:tree.V[0])tree.dfs2(x, 0);
		for(int i=0;i<n;i++){
			for(int j=0;j<=112;j++)
				sum[i][j]+=1.0*tree.f[i][j]/tree.sz[0][j];
		}
		/*
		for(int i=0;i<1000000;i++){
			int x=R()%nn;
			int y=R()%nn;
			for(int f1=0;f1<112;f1++)if(flag[x][f1])
				for(int f2=0;f2<112;f2++)if(flag[y][f2]){
					sum_dis[f1][f2]+=tree.dis(x, y);
					cnt_dis[f1][f2]++;
				}
		}
		*/
	}
	/*
	for(int f1=0;f1<112;f1++)
		for(int f2=0;f2<112;f2++)
			printf("%.15f%c", cnt_dis[f1][f2]?sum_dis[f1][f2]/cnt_dis[f1][f2]:0.0, f2==111?'\n':' ');
	return 0;*/
	n--;
	freopen("proteins/node_distance.txt","w",stdout);
	for(int i=0;i<n;i++)
		for(int j=0;j<=112;j++)
			printf("%.15f\n",sum[i][j]);
	fclose(stdout);
}
