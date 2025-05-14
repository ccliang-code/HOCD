function output = TCIMF_LCMV_HOCDver(HIM,indexes,options,gt,nys_flag,sr,ordered_class,k_th,group_k1,currentclass_indexes,preclsr_indexes)
this_class = ordered_class(k_th);
% [Bnds,NumOfClasses]=size(signature);
NumOfClasses = max(gt);
[m,n,o]=size(HIM);

X=reshape(HIM,m*n,o);

if nys_flag == 0
    K = constructKernel(X,[],options);
    K_hat = K^2;
    E_hat = K;
else
    [K_hat,E_hat] = KM_sub_HOCDver(HIM, options,sr,gt,currentclass_indexes,preclsr_indexes);
end

% Compress data
% [K,Y] = CS_KMD(HIM, options,sr,gt);
% [K,Y] = CS_KMD_rand(HIM, options,sr,gt);
% [K,Y] = KM_nys_cs(HIM, options,sr,gt);

% A2 = dense_mat(floor(N*0.6), N);

% Y = A*K;
% K = A*K*A';

% Y_signature = generate_d_index(Y', indexes , gt);
% invK = inv(K);
% clear A
% 

% for k = 1:NumOfClasses
%     Y_target = Y_signature(k,:);      
%     delta = Y'*invK*Y_target'/(Y_target*invK*Y_target');
%     output(:,:,k)=reshape(abs(delta),m,n);
% end
% Generate Eigenvalue and vector for correlation matrix
% tic
% if K ==K'
%     aa=1
% else
%     aa=0
% end
%%
%directly do inverse
%{

Y_signature = generate_d_index2d(Y', indexes , gt);
c = eye(size(Y_signature,1));
inv_K = inv(K);
% inv_K_2 = inv_K*inv_K;
 inv_K_2 = inv_K;

delta = Y'*inv_K_2*Y_signature'*inv(Y_signature*inv_K_2*Y_signature')*c;
% for k = 1:NumOfClasses
%     Y_target = Y_signature(k,:);       
%     delta = Y'*inv_K_2*Y_target'*inv(Y_target*inv_K_2*Y_target');
%     output(:,:,k)=reshape(abs(delta),m,n);
% end

output = reshape(delta,m,n,NumOfClasses );

%}

%%
%SVD solving inverse
%

disp('eig start')
[eigvector, eigvalue] = cal_corr(K_hat);
disp(strcat('eig=',string(length(eigvalue))))

% timeElapsed = toc
clear K_hat

% %
num_of_eigenvalues=size(eigvalue);

eigvalue=diag(eigvalue);

eig_inv=inv(eigvalue);
Y=E_hat'*eigvector;

reg_d = generate_d_index2d(Y, indexes , gt);
% Generate U
U = [];
if ~isempty(group_k1)
    for x = 1:length(group_k1)
        reg_class = group_k1(x);        
        m_hat = reg_d(reg_class,:);
        U = [U m_hat'];%
    end
end
d = reg_d(this_class,:)';
Y_signature = [d U]';
if isempty(U)
    c = [1];
else
%     c = [1; zeros(size(U,2),1)];
    c = eye(size(U,2)+1);
end
% for k = 1:NumOfClasses
%     Y_target = Y_signature(k,:);      
%     delta = Y*eig_inv*Y_target'/(Y_target*eig_inv*Y_target'); %KCEM
%     output(:,:,k)=reshape(abs(delta),m,n);
% end

%
% c = eye(size(Y_signature,1));
delta = Y*eig_inv*Y_signature'*inv(Y_signature*eig_inv*Y_signature')*c; %KLCMV
% output = reshape(delta,m,n,);
output = reshape(delta,m,n,NumOfClasses);
%}
