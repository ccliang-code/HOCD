function [K,Y] = KM_sub_HOCDver(HIM, options,sr_ran,gt,currentclass_indexes,preclsr_indexes)

[m,n,b] = size(HIM);
X=reshape(HIM,m*n,b);
N = length(X);
% N = length(X_gt);
M = floor(N*sr_ran);

Y = [];
r = n;
count = 0;

%% Random sampling
% ind_rand = randperm(N);
% X_r = X(ind_rand(1:floor(M)),:) ;
%% clustering
disp('cluster start')
idx_rmv = zeros(size(X,1),1);
X_rmvC = X;
if isempty(preclsr_indexes)
    idx_rmv(currentclass_indexes(:),1) = 1;
    X_rmvC(idx_rmv==1,:) = [];
else
    idx_rmv(currentclass_indexes(:),1) = 1;
    idx_rmv(preclsr_indexes(:),1) = 1;
    X_rmvC(idx_rmv==1,:) = [];
end
[idx, X_r] = eff_kmeans(X_rmvC, floor(sr_ran*N), 15);
disp('KM start')
%%
% for i = 0 : n/r-1
%         [K,~] = constructKernel(X_r,X(i*r*m+1:i*r*m+r*m,: ),options);
%         Y = [Y, K];
% end
Y = constructKernel(X_r,X,options);
K = Y*Y';