function out = SID(cutba,target)
p=target./(sum(target.^2))^(1/2);
[m,n]=size(cutba);
SIDmatrix=zeros(m,1);
for i=1:m
        q1=cutba(i,:);
        q=q1./(sum(q1.^2))^(1/2);
        D1=sum(p.*log(p/q));
        D2=sum(q.*log(q/p));
        SIDmatrix(i,1)=abs(D1+D2);
end
out=SIDmatrix;

return;