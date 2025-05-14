function [eigvector, eigvalue] = cal_corr(K)

[eigvector, eigvalue] = eig(K);
% tic
% [eigvector, eigvalue] = eigs(K,1000);
% toc
clear K
eigvalue = diag(eigvalue);    
[junk, index] = sort(-eigvalue);
eigvalue = eigvalue(index);
eigvector = eigvector(:,index);


maxEigValue = max(abs(eigvalue));
eigIdx = find(abs(eigvalue)/maxEigValue < 1e-11);
eigvalue (eigIdx) = [];
eigvector (:,eigIdx) = [];

% for i=1:length(eigvalue) % normalizing eigenvector
%     eigvector(:,i)=eigvector(:,i)/sqrt(eigvalue(i));
% end
