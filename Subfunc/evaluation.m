function [PA,MS,PR,POA,PAA,POPR,PAPR,PR_noBKG,POPR_noBKG,PAPR_noBKG] = evaluation(result, groundtruth)
% resize
[x1 y1] = size(result);
if y1~=1
    res1 = reshape(result,[x1*y1, 1]);
else
    res1 = result;
end
[x2 y2] = size(groundtruth);
if y2~=1
    gt1 = reshape(groundtruth,[x2*y2, 1]);
else
    gt1 = groundtruth;
end
% Create confusion matrix
reg = confusionmat(res1, gt1);
class_no = max(gt1);
% Move BKG
con_mat = reg;
con_mat(end+1,:) = reg(1,:);
con_mat(:,end+1) = con_mat(:,1);
con_matrix = con_mat(2:end,2:end);
% Accuracy & MS rate
for i = 1:length(con_matrix)-1
    PA(i,1) = con_matrix(i,i)/sum(con_matrix(:,i));
    MS(i,1) = 1 - PA(i);
    PR(i,1) = con_matrix(i,i)/sum(con_matrix(i,:));
    PCj(i,1) = length(find(gt1==i))/sum(gt1>0); 
end
%If NaN then 0
PR(isnan(PR))=0;
result_noBKG = res1;
result_noBKG(gt1==0) = [];
for i = 1:class_no
    p_c_hat(i,1) =  sum(res1==i)/length(gt1);
    p_c_hat_nobkg(i,1) = sum(result_noBKG==i)/sum(gt1>0); % No BKG
end
% OA 
POA = PCj'*PA;
% AA
% PAA = sum(PA)/(length(con_matrix)-1);
PAA = mean(PA);
POPR = p_c_hat'*PR;
PAPR = mean(PR);
%% No BKG
gt_noBKG = gt1;
gt_noBKG(gt1==0) = [];
comat_noBKG = confusionmat(result_noBKG, gt_noBKG);
for i = 1: size(comat_noBKG,1)
    PR_noBKG(i,1) = comat_noBKG(i,i)/sum(comat_noBKG(i,:));
end
%If NaN then 0
PR_noBKG(isnan(PR_noBKG))=0;
POPR_noBKG = p_c_hat_nobkg'*PR_noBKG;
PAPR_noBKG = mean(PR_noBKG);
% 123