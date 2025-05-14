function [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(cem_bw,GT,w_ts)
class_num = max(max(GT));
[a,b,c] = size(cem_bw);
new_cembw = nan(a,b,class_num);
if c==1
    for i = 1:class_num
        new_cembw(:,:,i) = (cem_bw==i);
    end   
    clear cem_bw
    cem_bw = new_cembw;
end
[x,y] = size(GT);
gt = reshape(GT, x*y, 1);
BKG = find(gt == 0);
w_b = length(BKG)/length(gt);
for i = 1: class_num
    N_ts(i) = length(find(GT == i)); 
end
bw2d = reshape(cem_bw, x*y, class_num);
bw2d_sum = sum(bw2d,2);
b_result = find(bw2d_sum == 0);
bw2d_bkg = bw2d;
bw2d_bkg(find(gt==0),:)=0;
% bw2d_bkg_sum = sum(bw2d_bkg,2);
N_hat= [];
N_hat_bkg = [];
for i = 1:class_num  
    gt = reshape(double(GT==i), x*y, 1);
    g_result = find(gt == 1);
    d_result = find(bw2d(:,i) == 1);
    n_hat{i} = d_result;
    N_hat = union(N_hat,d_result);
    Pa(i,1) = length(intersect(d_result,g_result))/length(g_result); 
    Pr(i,1) = length(intersect(d_result,g_result))/length(d_result);
    PF(i,1) = length(intersect(d_result,BKG))/length(BKG);
    d_bkg_result = find(bw2d_bkg(:,i) == 1);
    Pr_b(i,1) = length(intersect(d_bkg_result,g_result))/length(d_bkg_result);
    n_hat_bkg{i} = d_bkg_result;
    N_hat_bkg = union(N_hat_bkg,d_bkg_result);
end

for i = 1:class_num
    p_c_hat(i) =  length(n_hat{i})/length(N_hat);
    p_c_hat_bkg(i) = length(n_hat_bkg{i})/length(N_hat_bkg);
end
Pa = round(Pa,4);
Pr = round(Pr,4);
Pr_b = round(Pr_b,4);
Pr(isnan(Pr))=0;
Pr_b(isnan(Pr_b))=0;
% When change proformance weight of each class N_ts to w_ts
Poa = round(w_ts*Pa,4);
Paa = round(mean(Pa),4);
Popr = p_c_hat*Pr;
Papr=round(mean(Pr),4);
Popr_b = p_c_hat_bkg*Pr_b;
Papr_b = round(mean(Pr_b),4);
Pbac = length(intersect(BKG,b_result))/length(BKG);
Pac = (1-w_b)*Poa + w_b*Pbac;