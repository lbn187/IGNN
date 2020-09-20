#include<bits/stdc++.h>
#define N 4444
#define M 1666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m, F[N];
int cnt[111];
double dis_list[1111];
bool vis[N];
int d[N][N],q[N],dd[N];
vector<int>V[N];
struct{
	int x,y;
	double z,d;
}train_pos[M], train_neg[M], valid_pos[M], valid_neg[M], test_pos[M], test_neg[M];
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
}tree[1111];
int main(){
	srand(time(0));
	n=4267;
	freopen("ddi/all.txt","r",stdin);
	scanf("%d",&train_pos_m);
	for(int i=0;i<train_pos_m;i++){
		int x,y;
		scanf("%d%d",&x, &y);
		train_pos[i].x = x;
		train_pos[i].y = y;
		V[x].push_back(y);
		V[y].push_back(x);
	}
	scanf("%d",&train_neg_m);
	for(int i=0;i<train_neg_m;i++)scanf("%d%d",&train_neg[i].x, &train_neg[i].y);
	scanf("%d",&valid_pos_m);
	for(int i=0;i<valid_pos_m;i++)scanf("%d%d",&valid_pos[i].x, &valid_pos[i].y);
	scanf("%d",&valid_neg_m);
	for(int i=0;i<valid_neg_m;i++)scanf("%d%d",&valid_neg[i].x, &valid_neg[i].y);
	scanf("%d",&test_pos_m);
	for(int i=0;i<test_pos_m;i++)scanf("%d%d",&test_pos[i].x, &test_pos[i].y);
	scanf("%d",&test_neg_m);
	for(int i=0;i<test_neg_m;i++)scanf("%d%d",&test_neg[i].x, &test_neg[i].y);
	for(int s=0;s<n;s++){
		printf("%d\n",s);
		int h=0, t=0;
		q[++t]=s;
		for(int i=0;i<n;i++)vis[i]=0;
		vis[s]=1;
		for(;h^t;){
			int x=q[++h];
			for(int y:V[x])if(!vis[y]){
				vis[y]=1;
				d[s][y]=d[s][x]+1;
				q[++t]=y;
			}
		}
	}
	for(int i=0;i<train_pos_m;i++){
		printf("E%d\n",i);
		int st=train_pos[i].x, ed=train_pos[i].y;
		int h=0, t=0;
		q[++t]=st;
		for(int i=0;i<n;i++)vis[i]=0, dd[i]=-1;
		vis[st]=1;
		dd[st]=0;
		for(;h^t;){
			int x=q[++h];
			if(vis[ed])break;
			for(int y:V[x])if(!vis[y]){
				if(x==st&&y==ed)continue;
				vis[y]=1;
				dd[y]=dd[x]+1;
				q[++t]=y;
			}
		}
		d[st][ed]=d[ed][st]=dd[ed];
	}
	fclose(stdin);
	freopen("ddi/train_pos_mindis.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%d\n",d[train_pos[i].x][train_pos[i].y]);
	fclose(stdout);
	freopen("ddi/train_neg_mindis.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)printf("%d\n",d[train_neg[i].x][train_neg[i].y]);
	fclose(stdout);
	freopen("ddi/valid_pos_mindis.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%d\n",d[valid_pos[i].x][valid_pos[i].y]);
	fclose(stdout);
	freopen("ddi/valid_neg_mindis.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%d\n",d[valid_neg[i].x][valid_neg[i].y]);
	fclose(stdout);
	freopen("ddi/test_pos_mindis.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%d\n",d[test_pos[i].x][test_pos[i].y]);
	fclose(stdout);
	freopen("ddi/test_neg_mindis.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%d\n",d[test_neg[i].x][test_neg[i].y]);
	fclose(stdout);
}
