#include<bits/stdc++.h>
#define N 588888
#define M 31111111
#define MM 6666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
int vis[N];
int q[M],dd[N];
int F[N];
vector<int>V[N];
struct{
	int x,y,z;
}train_pos[M], train_neg[M], valid_pos[MM], valid_neg[MM], test_pos[MM], test_neg[MM];
int R(){
	int x=rand()%32768;
	int y=rand()%32768;
	return x*32768+y;
}
int main(){
	srand(time(0));
	n=576289;
	freopen("ppa/all.txt","r",stdin);
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
	fclose(stdin);
	int K=1000;
	for(int i=0;i<n;i++)vis[i]=-1;
	for(int i=0;i<K;i++){
		printf("%d\n",i);
		int st=R()%n;
		for(int j=0;j<n;j++)dd[j]=40;
		int h=0,t=0;
		q[++t]=st;
		vis[st]=i;
		dd[st]=0;
		for(;h<t;){
			int x=q[++h];
			for(int y:V[x])if(vis[y]!=i){
				vis[y]=i;
				dd[y]=dd[x]+1;
				q[++t]=y;
			}
		}
		for(int j=0;j<train_pos_m;j++)train_pos[j].z+=dd[train_pos[j].x]+dd[train_pos[j].y];
		for(int j=0;j<train_neg_m;j++)train_neg[j].z+=dd[train_neg[j].x]+dd[train_neg[j].y];
		for(int j=0;j<valid_pos_m;j++)valid_pos[j].z+=dd[valid_pos[j].x]+dd[valid_pos[j].y];
		for(int j=0;j<valid_neg_m;j++)valid_neg[j].z+=dd[valid_neg[j].x]+dd[valid_neg[j].y];
		for(int j=0;j<test_pos_m;j++)test_pos[j].z+=dd[test_pos[j].x]+dd[test_pos[j].y];
		for(int j=0;j<test_neg_m;j++)test_neg[j].z+=dd[test_neg[j].x]+dd[test_neg[j].y];
	}
	freopen("ppa/train_pos_anchordis.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%d\n",train_pos[i].z);
	fclose(stdout);
	freopen("ppa/train_neg_anchordis.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)printf("%d\n",train_neg[i].z);
	fclose(stdout);
	freopen("ppa/valid_pos_anchordis.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%d\n",valid_pos[i].z);
	fclose(stdout);
	freopen("ppa/valid_neg_anchordis.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%d\n",valid_neg[i].z);
	fclose(stdout);
	freopen("ppa/test_pos_anchordis.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%d\n",test_pos[i].z);
	fclose(stdout);
	freopen("ppa/test_neg_anchordis.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%d\n",test_neg[i].z);
	fclose(stdout);
}
