clear;clc;
%%%%%%Example 1%%%%%%%%%%%
X1=rand(30,2);
X2=ones(30,1)*[1,2]+rand(30,2);
X3=ones(30,1)*[3,1]+rand(30,2);
X=[X1;X2;X3];
Y=[ones(30,1);2*ones(30,1);3*ones(30,1)];
k=max(Y);knn=1;g1=1;g2=1;c1=1;c2=1;t1=0.3;t2=0.7;
iY=NNG(X,k,knn);
[pY,wt]=ADPC(X,iY,g1,g2,c1,c2,t1,t2);
Accuracy=macc(Y,pY)

%%%%%%Example 2%%%%%%%%%%%
% load iris
% X=iris(:,1:4);Y=iris(:,5);
% k=max(Y);knn=1;cw=2;cb=2;c1=0.0625;c2=0.0625;t1=0.9;t2=1;
% iY=NNG(X,k,knn);
% [pY,wt]=ADPC(X,iY,cw,cb,c1,c2,t1,t2);
% Accuracy=macc(Y,pY)