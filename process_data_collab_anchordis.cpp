#include<bits/stdc++.h>
#define N 244444
#define M 1666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
int vis[N];
int q[M],dd[N];
int F[N];
vector<int>V[N];
struct{
	int x,y,z;
}train_pos[M], train_neg[M], valid_pos[M], valid_neg[M], test_pos[M], test_neg[M];
int R(){
	int x=rand()%32768;
	int y=rand()%32768;
	return x*32768+y;
}
int main(){
	srand(time(0));
	n=235868;
	freopen("collab/all.txt","r",stdin);
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
		for(int j=0;j<n;j++)dd[j]=15;
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
		int cnt1=0, cnt2=0, tmp1=0, tmp2=0;
		for(int j=0;j<train_pos_m;j++){
			if(dd[train_pos[j].x]==15&&dd[train_pos[j].y]==15)train_pos[j].z+=0;
			else if(dd[train_pos[j].x]==15)train_pos[j].z+=0/dd[train_pos[j].y];
			else if(dd[train_pos[j].y]==15)train_pos[j].z+=0/dd[train_pos[j].x];
			else train_pos[j].z+=dd[train_pos[j].x]+dd[train_pos[j].y], tmp1++;
			if(i==0)cnt1+=train_pos[j].z;
		}
		for(int j=0;j<train_neg_m;j++){
			if(dd[train_neg[j].x]==15&&dd[train_neg[j].y]==15)train_neg[j].z+=0;
			else if(dd[train_neg[j].x]==15)train_neg[j].z+=0/dd[train_neg[j].y];
			else if(dd[train_neg[j].y]==15)train_neg[j].z+=0/dd[train_neg[j].x];
			else train_neg[j].z+=dd[train_neg[j].x]+dd[train_neg[j].y], tmp2++;
			if(i==0)cnt2+=train_neg[j].z;
		}
		if(i==0)printf("%.15lf %.15lf\n",1.0*cnt1/tmp1, 1.0*cnt2/tmp2);
		cnt1=cnt2=0;tmp1=tmp2=0;
		for(int j=0;j<valid_pos_m;j++){
			if(dd[valid_pos[j].x]==15&&dd[valid_neg[j].y]==15)valid_pos[j].z+=0;
			else if(dd[valid_pos[j].x]==15)valid_pos[j].z+=0/dd[valid_pos[j].y];
			else if(dd[valid_pos[j].y]==15)valid_pos[j].z+=0/dd[valid_pos[j].x];
			else valid_pos[j].z+=dd[valid_pos[j].x]+dd[valid_pos[j].y], tmp1++;
			if(i==0)cnt1+=valid_pos[j].z;
		}
		for(int j=0;j<valid_neg_m;j++){
			if(dd[valid_neg[j].x]==15&&dd[valid_neg[j].y]==15)valid_neg[j].z+=0;
			else if(dd[valid_neg[j].x]==15)valid_neg[j].z+=dd[valid_neg[j].y];
			else if(dd[valid_neg[j].y]==15)valid_neg[j].z+=dd[valid_neg[j].x];
			else valid_neg[j].z+=dd[valid_neg[j].x]+dd[valid_neg[j].y], tmp2++;
			if(i==0)cnt2+=valid_neg[j].z;
		}
		if(i==0)printf("%.15lf %.15lf\n",1.0*cnt1/tmp1, 1.0*cnt2/tmp2);
		cnt1=cnt2=0;tmp1=tmp2=0;
		for(int j=0;j<test_pos_m;j++){
			if(dd[test_pos[j].x]==15&&dd[test_pos[j].y]==15)test_pos[j].z+=1;
			else if(dd[test_pos[j].x]==15)test_pos[j].z+=dd[test_pos[j].y]*2;
			else if(dd[test_pos[j].y]==15)test_pos[j].z+=dd[test_pos[j].x]*2;
			else test_pos[j].z+=dd[test_pos[j].x]+dd[test_pos[j].y];
		}
		for(int j=0;j<test_neg_m;j++){
			if(dd[test_neg[j].x]==15&&dd[test_neg[j].y]==15)test_neg[j].z+=1;
			else if(dd[test_neg[j].x]==15)test_neg[j].z+=dd[test_neg[j].y]*2;
			else if(dd[test_neg[j].y]==15)test_neg[j].z+=dd[test_neg[j].x]*2;
			else test_neg[j].z+=dd[test_neg[j].x]+dd[test_neg[j].y];
		}
	}
	freopen("collab/train_pos_anchordis.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%d\n",train_pos[i].z);
	fclose(stdout);
	freopen("collab/train_neg_anchordis.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)printf("%d\n",train_neg[i].z);
	fclose(stdout);
	freopen("collab/valid_pos_anchordis.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%d\n",valid_pos[i].z);
	fclose(stdout);
	freopen("collab/valid_neg_anchordis.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%d\n",valid_neg[i].z);
	fclose(stdout);
	freopen("collab/test_pos_anchordis.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%d\n",test_pos[i].z);
	fclose(stdout);
	freopen("collab/test_neg_anchordis.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%d\n",test_neg[i].z);
	fclose(stdout);
}
