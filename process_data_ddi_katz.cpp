#include<bits/stdc++.h>
#define N 4288
#define M 1666666
using namespace std;
int n, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
int dp[N][N],dp_nxt[N][N];
double score[N][N];
vector<int>V[N];
struct{
	int x,y;
	double z,d;
}train_pos[M], train_neg[M], valid_pos[M], valid_neg[M], test_pos[M], test_neg[M];
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
	for(int i=0;i<n;i++)dp[i][i]=1;
	double beta=0.1, tmp=1.0;
	for(int step=1;step=20;step++){
		tmp=tmp*beta;
		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++){
				for(int x:V[j]){
					dp_nxt[i][x]+=dp[i][j];
				}
			}
		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++){
				dp[i][j]=dp_nxt[i][j];
				dp_nxt[i][j]=0;
				score[i][j]+=tmp*dp[i][j];
			}
	}
	fclose(stdin);
	freopen("ddi/train_pos_katz.txt","w",stdout);
	for(int i=0;i<train_pos_m;i++)printf("%.15lf\n",score[train_pos[i].x][train_pos[i].y]);
	fclose(stdout);
	freopen("ddi/train_neg_katz.txt","w",stdout);
	for(int i=0;i<train_neg_m;i++)printf("%.15lf\n",score[train_neg[i].x][train_neg[i].y]);
	fclose(stdout);
	freopen("ddi/valid_pos_katz.txt","w",stdout);
	for(int i=0;i<valid_pos_m;i++)printf("%.15lf\n",score[valid_pos[i].x][valid_pos[i].y]);
	fclose(stdout);
	freopen("ddi/valid_neg_katz.txt","w",stdout);
	for(int i=0;i<valid_neg_m;i++)printf("%.15lf\n",score[valid_neg[i].x][valid_neg[i].y]);
	fclose(stdout);
	freopen("ddi/test_pos_katz.txt","w",stdout);
	for(int i=0;i<test_pos_m;i++)printf("%.15lf\n",score[test_pos[i].x][test_pos[i].y]);
	fclose(stdout);
	freopen("ddi/test_neg_katz.txt","w",stdout);
	for(int i=0;i<test_neg_m;i++)printf("%.15lf\n",score[test_neg[i].x][test_neg[i].y]);
	fclose(stdout);
}
