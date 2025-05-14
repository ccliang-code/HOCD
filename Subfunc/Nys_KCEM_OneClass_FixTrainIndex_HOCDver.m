function output = Nys_KCEM_OneClass_FixTrainIndex_HOCDver(HIM, options, gt, nys_flag, sr, class_th, currentclass_indexes,preclsr_indexes)
% [Bnds,NumOfClasses]=size(signature);
NumOfClasses = max(gt);
[m,n,o]=size(HIM);

X = reshape(HIM,m*n,o);

if nys_flag == 0
    K = constructKernel(X,[],options);
    K_hat = K^2;
    E_hat = K;
else
    [K_hat,E_hat] = KM_sub_HOCDver(HIM, options,sr,gt,currentclass_indexes,preclsr_indexes);
end
%%
%SVD solving inverse
%
disp('eig start')
if sum(sum(isnan(K_hat)))>0
    123
    
end
[eigvector, eigvalue] = cal_corr(K_hat);
disp(strcat('eig = ',string(length(eigvalue))))

% timeElapsed = toc
clear K_hat

% %
num_of_eigenvalues = size(eigvalue);

eigvalue = diag(eigvalue);

eig_inv = inv(eigvalue);
Y = E_hat'*eigvector;
%% KCEM
Y_signature = generate_d_index2d(Y, currentclass_indexes , gt);     
Y_target = Y_signature(class_th,:);      
delta = Y*eig_inv*Y_target'/(Y_target*eig_inv*Y_target'); %KCEM
output = reshape(abs(delta),m,n);

