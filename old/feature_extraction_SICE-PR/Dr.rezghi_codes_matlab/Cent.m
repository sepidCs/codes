function ran=Cent(X,T)
[m,n]= size(X);
for i=1: fix(n/T)
    Temp=X(:,1+(i-1)*T:i*T);
    nTemp=normr(Temp);
    temp2=abs(nTemp*nTemp');
    temp2=temp2-diag(diag(temp2));
    S(:,:,i)=temp2;
end
S=tensor(S);
%% H_matrix 
% reshape
SM=tenmat(S,[3],[1 2]);
SM1=double(SM)';
%list of norm 1 of each row
lm=vecnorm(SM1,1,2);
% normalize rows ....
nSM1T=normr(SM1);

%replace rows with zero norm 1 to 0 ...redandant?
nSM1T(lm==0,:)=0;

GSim=nSM1T*nSM1T';
GSim=GSim-diag(diag(GSim));
% replace zero rows with 1/size 
T=vecnorm(GSim,1,1);
dd=zeros(size(T'));
dd(T==0)=1;
%why 
GSim=GSim+ones(size(GSim,1),1)*dd'/size(GSim,1);

Nor = 1./vecnorm(GSim,1,1);
for i=1:size(GSim,1)
    GSim(:,i)=GSim(:,i).*Nor(i);
end
%%
e=ones(size(GSim,1),1);
A=0.85*GSim+(1-0.85)*e*e'/size(GSim,1);

[U,~]=eig(A);
p=abs(U(:,1));
%%
p(p<mean(p))=0;
P=reshape(p,[m,m]);
[row,col,val]=find(sparse(P));
ran=zeros(m,1);
for i=1:m
    ran(i)=sum(val(row==i))+sum(val(col==i));
end

