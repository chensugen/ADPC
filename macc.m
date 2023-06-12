 function Acc=macc(Y,pY)
% Y is truth label;
% pY is prediction label;
[m,~]=size(Y);
for i=1:m
    for j=1:m
       a=(Y(i)==Y(j));
        M(i,j)=a;
    end
end
for i=1:m
    for j=1:m
        b=(pY(i)==pY(j));
        N(i,j)=b;
    end
end
t=M+N==2;
r=M+N==0;
n11=sum(t(:));
n00=sum(r(:));
Acc=(n11+n00-m)/(m^2-m)*100;
