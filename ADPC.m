function [Y,wt]= ADPC(X,iY,cw,cb,c1,c2,t1,t2)
%ADPC without norm and use wt as the initialization
%Input£ºX : Data Matrix.Each row vector of fea is a data point
%       iY : the initial label obtained by NNG
%       Parameters : cw,cb,c1,c2,t1,t2
%Output: Y : Predict the class of X
%        wt : Predict the plane vector
%Examples:
% X1=rand(30,2);
% X2=ones(30,1)*[1,2]+rand(30,2);
% X3=ones(30,1)*[3,1]+rand(30,2);
% X=[X1;X2;X3];
% Y=[ones(30,1);2*ones(30,1);3*ones(30,1)];
% k=max(Y);knn=1;g1=1;g2=1;c1=1;c2=1;t1=0.3;t2=0.7;
% iY=NNG(X,k,knn);
% [pY,wt]=ADPC(X,iY,g1,g2,c1,c2,t1,t2);
% Accuracy=macc(Y,pY)
som=1;
ite=0;
Y=iY;
wt=[];
while som>1e-3 && ite<100
             ite=ite+1;
             [pY,wt]= IteOne([X,ones(size(X,1),1)],Y,cw,cb,c1,c2,t1,t2,wt);
             som=norm(pY-Y,1);
             Y=pY;
end
end

function [pY,wt]= IteOne(X,Y,c1,c2,c3,c4,t1,t2,wt)
%function w= IteOne(X,Y,c1,c2,c3,c4)
% X is data with ones as its last column;
% Y is getted by Some Initialization.m
% Y must be 1,2,3,.... , cannot be blank of any number.
delta=0.3;
s=-0.2;
num=max(Y);
n=size(X,2);
totalw=[];
for i=1:num
    X1=X(Y==i,:);
    X2=X(Y~=i,:);
    if isempty(wt)
        w0=FirstStep(X1);    
    else
        w0=wt(:,i);
    end
    m1=size(X1,1);
    ite=0;
    som=1;
    G=eye(n)+X1'*(c2/m1*eye(m1)+(c1-c2)/m1/m1*ones(m1,m1))*X1;
    wk0=w0;
    wk1=w0;
    wkbest=[];
    Fbest=inf;
    theta0=1;
    theta1=1;
    L=(t1+t2)*norm(c3*sum(X1,1)+c4*sum(X2,1));
    while ite<100 && som>1e-3
        ite=ite+1;
        if mod(ite,100)==0
            theta1=1;
        end
        beta=(theta0-1)/theta1;
        theta0=theta1;
        theta1=(1+sqrt(1+4*theta0^2))/2; 
        u=wk1+beta*(wk1-wk0);
        wk0=wk1;
        ttmp=X1*wk1;        
        tmp1=c3*X1'*(-t1*sign(max(-t1*(-s+2-delta+ttmp),0))+t2*sign(max(t2*(s-2+delta+ttmp),0))); %gradeint of P2-1
        ttmp=X2*wk1;
        tmp2=c4*X2'*(-t2*sign(max(t2*(s-ttmp),0))+t1*sign(max(t1*(s+ttmp),0)));  %gradeint of P2-2
        ttmp=X1*u;        
        tmp3=c3*X1'*(-t1*sign(max(-t1*(1-delta+ttmp),0))+t2*sign(max(t2*(-1+delta+ttmp),0)));  %gradeint of f-1
        ttmp=X2*u;        
        tmp4=c4*X2'*(-t2*sign(max(t2*(1+delta-ttmp),0))+t1*sign(max(t1*(1+delta+ttmp),0)));  %gradeint of f-2
        wk1=(L*eye(n)+G)\(tmp1+tmp2-tmp3-tmp4+L*u);
        Ftmp=F(X1,X2,G,wk1,c3,c4,delta,s,t1,t2);
        if Ftmp<Fbest
            Fbest=Ftmp;
            wkbest=wk1;
        end
        som=norm(wk1-wk0);
    end
    if norm(wk1)>0
        totalw=[totalw,wkbest];
    end
end

if ~isempty(totalw)
    [~,pY]=min(abs(X*totalw),[],2);
else
    pY=ones(size(X,1),1);
end
wt=totalw;
end

function u=FirstStep(A)
% compute: min ||Aw||, s.t. ||w(1:n-1)||=1.
% u=[w;b]
[m,n]=size(A);
H=A(:,1:n-1)'*(1/m*ones(m,m)-eye(m))*A(:,1:n-1);
[V,D]=eig((H+H')/2);
[~,t]=min(abs(diag(D)));
w=V(:,t);
b=-1/m*sum(A(:,1:n-1),1)*w;
u=[w;b];
end

function val=F(X1,X2,G,w,c3,c4,delta,s,t1,t2)
% objective function
val=0.5*w'*G*w;
tmp=X1*w;
valpos=tmp;
valpos(tmp<=-2+s+delta )=-t1*(s-1);
valpos(tmp>=2-delta-s)=t2*(1-s);
valpos(tmp>-2+s+delta & tmp <-1+delta)=-t1*(valpos(tmp>-2+s+delta & tmp <-1+delta)+1-delta);
valpos(tmp>=-1+delta & tmp<=1-delta)=0;
valpos(tmp>1-delta & tmp<2-s-delta)=t2*(valpos(tmp>1-delta & tmp<2-s-delta)-1+delta);
val=val+c3*sum(valpos);
tmp=X2*w;
valpos=tmp;
valpos(tmp<=-1-delta)=-t1-t1*delta-t1*s+2+2*delta;
valpos(tmp>=1+delta)=-t2-t2*delta+2+2*delta-t2*s;
valpos(tmp>-1-delta & tmp<s)=t1*valpos(tmp>-1-delta & tmp<s)+2+2*delta-t1*s;
valpos(tmp>=s & tmp<=-s)=2+2*delta;
valpos(tmp>-s & tmp<1+delta)=-t2*valpos(tmp>-s & tmp<1+delta)+2+2*delta-t2*s;
val=val+c4*sum(valpos);
end