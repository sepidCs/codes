clear
clc
win=50;
all_data=10000;
test_train_split=0.9;
%% read data
Xdata = readtable('dataframes.csv');
X=cell2mat(table2cell(Xdata(:,2:end)));
% X=X(1:all_data,:)';
X=X';
%%

ran=Cent(X(1:(end-1),:),win);
[l,s]=sort(ran,'descend');
%%
X_without=X(1:10,:);
xp=mean(X_without,2);
Xp=X_without-kron(ones(1,size(X_without,2)),xp);
[U,S,V]=svds(Xp,10);
VV=S*V';
T=win
[m,n]= size(X);
for i=1:10
    DDtrain=X(s(1:i),1:(all_data*test_train_split))';
    Ddtest=X(s(1:i),(all_data*test_train_split+1):end)';
    ytrain=X(end,1:(all_data*test_train_split))';
    ytest1=X(end,(all_data*test_train_split+1):end)';
    treem=fitrtree(DDtrain,ytrain);
    ypm=predict(treem,Ddtest);
    R(i)=norm(ypm-ytest1)/norm(ytest1);
    %%
    DDtrain=VV(1:i,1:(all_data*test_train_split))';
    Ddtest=VV(1:i,(all_data*test_train_split+1):end)';
    treem=fitrtree(DDtrain,ytrain);
    ypm1=predict(treem,Ddtest);
    R1(i)=norm(ypm1-ytest1)/norm(ytest1);
end


plot(R,'*-')
hold on
plot(R1,'-0-')