#include<bits/stdc++.h>
#define N 255555
#define M 1666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m, F[N];
struct P{
	int x,y;
	double z,d;
};
bool cmp(P a,P b){
	return a.z<b.z;
}
double sum[N];
P train_pos[M], train_pos2[M], train_neg[M], valid_pos[M], valid_neg[M], test_pos[M], test_neg[M];
int gf(int x){
	if(F[x]==x)return x;
	return F[x]=gf(F[x]);
}
struct Tree{
	vector<int>V[N];
	int F[N][18];
	int d[N];
	int sz[N];
	long long f[N];
	int n;
	void ins(int x,int y){
		V[x].push_back(y);
		V[y].push_back(x);
	}
	void dfs(int x,int fa){
		for(int i=1;i<=17;i++)
			if(F[x][i-1]!=-1)
				F[x][i]=F[F[x][i-1]][i-1];
		sz[x]=1;
		for(int y:V[x])
			if(y!=fa){
				F[y][0]=x;
				d[y]=d[x]+1;
				dfs(y,x);
				sz[x]+=sz[y];
				f[x]+=f[y]+sz[y];
			}
	}
	void dfs2(int x,int fa){
		f[x]=f[fa]+n-2*sz[x];
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
	n=235868;
	freopen("collab/all.txt","r",stdin);
	scanf("%d",&train_pos_m);
	for(int i=0;i<train_pos_m;i++)scanf("%d%d",&train_pos[i].x, &train_pos[i].y), train_pos2[i] = train_pos[i];
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
	fclose(stdin);
	long long train_pos_sum = 0, train_neg_sum = 0, valid_pos_sum = 0, valid_neg_sum = 0, test_pos_sum = 0, test_neg_sum = 0;
	n++;
	for(int T=0;T<1000;T++){
		printf("%d\n",T);
		random_shuffle(train_pos2,train_pos2+train_pos_m);
		for(int i=0;i<n;i++)F[i]=i;
		for(int i=0;i<n;i++){
			tree.V[i].clear();
			tree.d[i]=0;
			tree.sz[i]=0;
			tree.f[i]=0;
			for(int j=0;j<=17;j++)tree.F[i][j]=-1;
		}
		int cnt=0;
		for(int i=0;i<train_pos_m;i++){
			int x=gf(train_pos2[i].x);
			int y=gf(train_pos2[i].y);
			if(x!=y){
				F[x]=y;
				tree.ins(train_pos2[i].x, train_pos2[i].y);
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
		for(int i=0;i<n;i++)sum[i]+=1.0*tree.f[i]/n/1000;
		for(int i=0;i<train_pos_m;i++){
			double z = tree.dis(train_pos[i].x, train_pos[i].y);
			train_pos[i].z += z;
			//train_pos_sum += z;
			train_pos[i].d += z * z;
		}
		for(int i=0;i<train_neg_m;i++){
			double z = tree.dis(train_neg[i].x, train_neg[i].y);
			train_neg[i].z += z;
			//train_neg_sum += z;
			train_neg[i].d += z * z;
		}
		for(int i=0;i<valid_pos_m;i++){
			double z = tree.dis(valid_pos[i].x, valid_pos[i].y);
			valid_pos[i].z += z;
			//valid_pos_sum += z;
			valid_pos[i].d += z*z;
		}
		for(int i=0;i<valid_neg_m;i++){
			double z = tree.dis(valid_neg[i].x, valid_neg[i].y);
			valid_neg[i].z += z;
			//valid_neg_sum += z;
			valid_neg[i].d += z*z;
		}
		for(int i=0;i<test_pos_m;i++){
			double z = tree.dis(test_pos[i].x, test_pos[i].y);
			test_pos[i].z += z;
			//test_pos_sum += z;
			test_pos[i].d += z*z;
		}
		for(int i=0;i<test_neg_m;i++){
			double z = tree.dis(test_neg[i].x, test_neg[i].y);
			test_neg[i].z += z;
			//test_neg_sum += z;
			test_neg[i].d += z*z;
		}
		/*
		for(int i=0;i<train_neg_m;i++)train_neg[i].z += tree.dis(train_neg[i].x, train_neg[i].y);
		for(int i=0;i<valid_pos_m;i++)valid_pos[i].z += tree.dis(valid_pos[i].x, valid_pos[i].y);
		for(int i=0;i<valid_neg_m;i++)valid_neg[i].z += tree.dis(valid_neg[i].x, valid_neg[i].y);
		for(int i=0;i<test_pos_m;i++)test_pos[i].z += tree.dis(test_pos[i].x, test_pos[i].y);
		for(int i=0;i<test_neg_m;i++)test_neg[i].z += tree.dis(test_neg[i].x, test_neg[i].y);
		*/
	}
	n--;
	freopen("collab/node_distance.txt","w",stdout);
	for(int i=0;i<n;i++)printf("%.15f\n",sum[i]);
	fclose(stdout);
	for(int i=0;i<train_pos_m;i++)train_pos[i].z/=1000, train_pos[i].d-=train_pos[i].z*train_pos[i].z;
	for(int i=0;i<train_neg_m;i++)train_neg[i].z/=1000, train_neg[i].d-=train_neg[i].z*train_neg[i].z;
	for(int i=0;i<valid_pos_m;i++)valid_pos[i].z/=1000, valid_pos[i].d-=valid_pos[i].z*valid_pos[i].z;
	for(int i=0;i<valid_neg_m;i++)valid_neg[i].z/=1000, valid_neg[i].d-=valid_neg[i].z*valid_neg[i].z;
	for(int i=0;i<test_pos_m;i++)test_pos[i].z/=1000, test_pos[i].d-=test_pos[i].z*test_pos[i].z;
	for(int i=0;i<test_neg_m;i++)test_neg[i].z/=1000, test_neg[i].d-=test_neg[i].z*test_neg[i].z;
	//for(int i=valid_pos_m - 1000; i<valid_pos_m; i++)printf("POS %.15f %.15f\n",valid_pos[i].z, valid_pos[i].d);
	//for(int i=0;i<1000;i++)printf("NEG %.15f %.15f\n",valid_neg[i].z, valid_neg[i].d);
	
	freopen("collab/train_pos_avg.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%.15f\n",train_pos[i].z);
	fclose(stdout);
	freopen("collab/train_neg_avg.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)printf("%.15f\n",train_neg[i].z);
	fclose(stdout);
	freopen("collab/valid_pos_avg.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%.15f\n",valid_pos[i].z);
	fclose(stdout);
	freopen("collab/valid_neg_avg.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%.15f\n",valid_neg[i].z);
	fclose(stdout);
	freopen("collab/test_pos_avg.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%.15f\n",test_pos[i].z);
	fclose(stdout);
	freopen("collab/test_neg_avg.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%.15f\n",test_neg[i].z);
	fclose(stdout);
	freopen("collab/train_pos_fangcha.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)
		if(train_pos[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(train_pos[i].d+1e-15));
	fclose(stdout);
	freopen("collab/train_neg_fangcha.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)
		if(train_neg[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(train_neg[i].d+1e-15));
	fclose(stdout);
	freopen("collab/valid_pos_fangcha.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)
		if(valid_pos[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(valid_pos[i].d+1e-15));
	fclose(stdout);
	freopen("collab/valid_neg_fangcha.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)
		if(valid_neg[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(valid_neg[i].d+1e-15));
	fclose(stdout);
	freopen("collab/test_pos_fangcha.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)
		if(test_pos[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(test_pos[i].d+1e-15));
	fclose(stdout);
	freopen("collab/test_neg_fangcha.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)
		if(test_neg[i].z < 1e-15)puts("0");else printf("%.15f\n",sqrt(test_neg[i].d+1e-15));
	fclose(stdout);


	/*
	freopen("ppa/train_pos_info.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum+=tree[T].dis(train_pos[i].x, train_pos[i].y);
		}
		printf("%.8f\n",sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(train_pos[i].x, train_pos[i].y));
	}
	fclose(stdout);
	freopen("ppa/train_neg_info.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum +=tree[T].dis(train_neg[i].x, train_neg[i].y);
		}
		printf("%.8f\n", sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(train_neg[i].x, train_neg[i].y));
	}
	fclose(stdout);
	freopen("ppa/valid_pos_info.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum+=tree[T].dis(valid_pos[i].x, valid_pos[i].y);
		}
		printf("%.8f\n",sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(valid_pos[i].x, valid_pos[i].y));
	}
	fclose(stdout);
	freopen("ppa/valid_neg_info.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum+=tree[T].dis(valid_neg[i].x, valid_neg[i].y);
		}
		printf("%.8f\n",sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(valid_neg[i].x, valid_neg[i].y));
	}
	fclose(stdout);
	freopen("ppa/test_pos_info.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum+=tree[T].dis(test_pos[i].x, test_pos[i].y);
		}
		printf("%.8f\n",sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(test_pos[i].x, test_pos[i].y));
	}
	fclose(stdout);
	freopen("ppa/test_neg_info.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++){
		double sum=0;
		for(int T=0;T<100;T++){
			sum+=tree[T].dis(test_neg[i].x, test_neg[i].y);
		}
		printf("%.8f\n",sum/100);
		//for(int T=0;T<100;T++)printf("%d\n",tree[T].dis(test_neg[i].x, test_neg[i].y));
	}
	fclose(stdout);*/
}
