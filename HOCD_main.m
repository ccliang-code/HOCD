clc;
clear all;
close all;
addpath('.\Subfunc\');
HSI = 1; % 1: Purdue; 2: Salinas; 3: PaviaU; 4: Houston
extime = 1;
TI_type = 1; % 1: Only one TI; 2: Different TI for each class
train_saple_rate = 1;
Similarity_threshold = 0.34;
CS_method = 2;%Similarity calculation methods: 1: SAM, 2: SID, 3: ED
classifier_type = 4;%1:P?R-IRTS-KCEM; 2:R-IRTS-KCEM; 3:P?-IRTS-KCEM; 4:P?R-IRTS-IKTCIMF; 5:R-IRTS-IKTCIMF; 6:P?-IRTS-IKTCIMF
T_count = 1;% Choose the way to setup the TI index
loop_limit = 1; % Maxima iteration times
%% TI setting
TI_set = 0.98* ones(1, 16);
%% Kernel parameter
options.KernelType = 'Gaussian';
%% Class classification priority(CCP)
CCP_type = 1; %1: Rinv; 2: OSP(Puper)
OSP_data = 3; %1: Classified result; 2: GT; 3: Classified result + Similarity using ClsR
nys_flag = 1;
hsi_folder = '.\DataSet\';
switch HSI
    case 1 %Purdue
        CS_threshold = Similarity_threshold;
        options.t = 2;%4;
        inputs = 'IndiaP';
        if TI_type == 1
            TI_setting = 0.98;
        else
            TI_setting = TI_set(T_count,:)';
            T_count = T_count+1;
        end
        hsi_name = 'Purdue';
        Ccv = 100;
        Gcv = 0.1250;
        sigma = 1.3;%1.3;
        n_sr = 0.13; %Nystorm sampling rate
        load([hsi_folder '\Indian_pines_gt.mat']);
        GT = indian_pines_gt;
        clear indian_pines_gt
    case 2 %Salinas
        CS_threshold = Similarity_threshold;%10;
        options.t = 2;%3;
        inputs = 'Salinas';
        if TI_type == 1
            TI_setting = 0.98;
        else
            TI_setting = TI_set(T_count,:)';
        end
        hsi_name = 'Salinas';
        Ccv = 100;
        Gcv = 0.250;
        sigma = 2;
        n_sr = 0.013; %Nystorm sampling rate
        load([hsi_folder '\Salinas_gt.mat']);
        GT = salinas_gt;
        clear salinas_gt
    case 3 %PaviaU
        CS_threshold = Similarity_threshold;
        options.t = 2;%5;
        inputs = 'PaviaU';
        if TI_type == 1
            TI_setting = 0.98;
        else
            TI_setting = TI_set(T_count,:)';
            T_count = T_count+1;
        end
        hsi_name = 'PaviaU';
        Ccv = 100;
        Gcv = 0.5;
        sigma = 1;
        n_sr = 0.007; %Nystorm sampling rate
        load([hsi_folder '\PaviaUniversity_gt.mat']);
        GT = double(paviaU_gt);
        clear paviaU_gt
    case 4 %Houston
        hsi_name = 'Houston';
        inputs = 'Houston';
        CS_threshold = Similarity_threshold;
        if TI_type == 1
            TI_setting = 0.98;
        else
            TI_setting = TI_set(T_count,:)';
            T_count = T_count+1;
        end
        options.t = 2;
        pca_band = 15;
        sigma = 1;
        n_sr = 0.002; %Nystorm sampling rate
        load([hsi_folder '\houston_gt.mat']);
        Ccv = 100;
        Gcv = 0.5;
end
%%
load ([hsi_folder '\' inputs '_normalized.mat']);
HIM = img;
class_no = max(max(GT));
sample_no = sum(sum(GT>0));

%% Generate training number for each class
[m,n,o] = size(HIM);
gt = reshape(GT,m*n,1);
HIM_2d = reshape(HIM,m*n,o);
%% Class probability
for i = 1: class_no
    hist(i) = length(find(GT == i));
    GT_class(:,:,i) = double(GT == i);
end
class_p = hist/sum(hist);

all_str_time = clock;
break_flag = 0;
for loop = 1:extime
    %% Load fixed training index
    load(['.\TrainingIndex_HOCC\' hsi_name '\trainingindex' num2str(train_saple_rate) '_' num2str(loop) '.mat']);%Load all_indexes
    train_label = indexes_label(:,1);
    total_no = size(indexes_label(:,1),1); %Total number of training sample
    para_set = struct('total_no',total_no,'n_sr',n_sr);
    str_time = clock;
    %% Calculate Class classification
    if CCP_type == 1
        % global sample correlation matrix R
        R = zeros(o);
        for i = 1:(m*n)
            r = HIM_2d(i,:);
            mtrx = r'*r;
            R = R + mtrx;
        end
        R = R/(m*n);
        % Sample mean
        class_mean = nan(class_no,o);
        P_ccp = nan(class_no,1);
        for i = 1:class_no
            reg = HIM_2d;
            reg((gt~=i),:) = [];
            class_mean(i,:) = mean(reg,1);
            clear reg
            P_ccp(i) = class_mean(i,:)*inv(R)*class_mean(i,:)';
        end
    else
        % Sample mean
        class_mean = nan(class_no,o);
        for i = 1:class_no
            reg = HIM_2d;
            reg((gt~=i),:) = [];
            class_mean(i,:) = mean(reg,1);
            clear reg
        end
        % Puper
        Im = eye(o);
        for i = 1:class_no
            Um = class_mean';
            Um(:,i)=[];
            Um_psudoinv = inv(Um'*Um)*Um';
            pUper = Im-(Um*Um_psudoinv);
            P_ccp(i) = class_mean(i,:)*pUper*class_mean(i,:)';
            clear Um
        end
        clear Im Um_psudoinv
    end
    [B,ordered_class] = sort(P_ccp,'descend');
    B = round(B,2);
    if OSP_data == 1 || OSP_data == 2
        %% Calculate class similarity
        cs_sam = nan(class_no);
        cs_ed = nan(class_no);
        cs_sid = nan(class_no);
        for i = 1:class_no
            for j = 1:class_no
                cs_sam(i,j) = SAM(class_mean(i,:),class_mean(j,:));
                cs_ed(i,j) = ED(class_mean(i,:),class_mean(j,:));
                cs_sid(i,j) = SID(class_mean(i,:),class_mean(j,:));
            end
        end
        %% Use threshold to determine the remove group
        group = cell(class_no-1,1);
        for group_loop = 1:class_no-1
            this_class = ordered_class(group_loop+1);
            reg_group = ordered_class(1:group_loop)';
            switch CS_method
                case 1
                    cs_name = 'SAM';
                    class_thidx = find(cs_sam(this_class,:)>=CS_threshold);
                case 2
                    cs_name = 'SID';
                    class_thidx = find(cs_sid(this_class,:)>=CS_threshold);
                case 3
                    cs_name = 'ED';
                    class_thidx = find(cs_ed(this_class,:)>=CS_threshold);
            end
            reg_group(~ismember(reg_group,class_thidx)) = [];
            group(group_loop,:) = {reg_group};
        end
    end
    %% Class probability
    cs_sam = zeros(class_no);
    cs_ed = zeros(class_no);
    cs_sid = zeros(class_no);
    result_otsu = nan(m,n,class_no);
    SF_map = nan(m,n,class_no);
    X_new = HIM;
    U = [];
    I = eye(o);
    saveTI = nan(loop_limit+1,class_no);
    savePA = nan(loop_limit+1,class_no);
    savePR = nan(loop_limit+1,class_no);
    clsR_indexes = cell(class_no, 1);
    for k = 1:class_no
        class_th = ordered_class(k);
        [m1,n1,o1] = size(X_new);
        X_new2d = reshape(X_new,m1*n1,o1);
        [X_new2d_reg,M,N] = scale_func(X_new2d);
        update_HIM = reshape(X_new2d_reg,m1,n1,o1);
        this_class_idx = all_indexes;
        this_class_idx(train_label ~= class_th,:) = [];
        %% TI_setting
        if TI_type == 1
            this_TIsetting = TI_setting;
        else
            this_TIsetting = TI_setting(class_th);
        end        
        %% Classifier
        switch classifier_type
            case 1 % P?R-IRTS-KCEM
                classifier_name = 'PuperR-IRTS-KCEM';
                TI = 0;
                % 0th
                indexes = this_class_idx(:,1)';
                if k==1
                    pre_clsr_indexes = [];
                else
                    pre_clsr_indexes = [];
                    rmv_class = group{k-1};
                    for q = 1:length(rmv_class)
                        pre_clsr_indexes = [pre_clsr_indexes;clsR_indexes_ccp{rmv_class(q),:}];
                    end
                    pre_clsr_indexes = pre_clsr_indexes';
                end
                cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex_HOCDver(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes,pre_clsr_indexes));
                cem_img = gaus_HIM(cem_img,sigma);
                threshold = graythresh(cem_img);
                old_map = imbinarize(cem_img,threshold);
                update_HIM(:,:,end+1) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex_HOCDver(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes,pre_clsr_indexes));
                    cem_img = gaus_HIM(cem_img,sigma);
                    threshold = graythresh(cem_img);
                    new_map = imbinarize(cem_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1) = cem_img;
                end
                clear update_HIM
            case 2 % R-IRTS-KCEM'
                classifier_name = 'R-IRTS-KCEM';
                TI = 0;
                % 0th
                indexes = this_class_idx(:,1)';
                if k==1
                    pre_clsr_indexes = [];
                else
                    pre_clsr_indexes = [];
                    rmv_class = group{k-1};
                    for q = 1:length(rmv_class)
                        pre_clsr_indexes = [pre_clsr_indexes;clsR_indexes_ccp{rmv_class(q),:}];
                    end
                    pre_clsr_indexes = pre_clsr_indexes';
                end
                cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex_HOCDver(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes,pre_clsr_indexes));
                cem_img = gaus_HIM(cem_img,sigma);
                threshold = graythresh(cem_img);
                old_map = imbinarize(cem_img,threshold);
                update_HIM(:,:,end+1) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex_HOCDver(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes,pre_clsr_indexes));
                    cem_img = gaus_HIM(cem_img,sigma);
                    threshold = graythresh(cem_img);
                    new_map = imbinarize(cem_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1) = cem_img;
                end
                clear update_HIM
            case 3 % P?-IRTS-KCEM
                classifier_name = 'Puper-IRTS-KCEM';
                TI = 0;
                % 0th
                indexes = this_class_idx(:,1)';
                cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes));
                cem_img = gaus_HIM(cem_img,sigma);
                threshold = graythresh(cem_img);
                old_map = imbinarize(cem_img,threshold);
                update_HIM(:,:,end+1) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(Nys_KCEM_OneClass_FixTrainIndex(update_HIM, options, gt, nys_flag, para_set.n_sr, class_th, indexes));
                    cem_img = gaus_HIM(cem_img,sigma);
                    threshold = graythresh(cem_img);
                    new_map = imbinarize(cem_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1) = cem_img;
                end
                clear update_HIM
            case 4 % P?R-IRTS-IKTCIMF
                classifier_name = 'PuperR-IRTS-IKTCIMF';
                TI = 0;
                indexes = this_class_idx(:,1)';
                if k==1
                    pre_clsr_indexes = [];
                else
                    pre_clsr_indexes = [];
                    rmv_class = group{k-1};
                    for q = 1:length(rmv_class)
                        pre_clsr_indexes = [pre_clsr_indexes;clsR_indexes_ccp{rmv_class(q),:}];
                    end
                    pre_clsr_indexes = pre_clsr_indexes';
                end
                group_tcimf = [1:class_no];
                group_tcimf(ordered_class(k)) = [];
                cem_img = abs(TCIMF_LCMV_HOCDver(update_HIM,all_indexes(:, 1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf, indexes,pre_clsr_indexes));
                cem_img = gaus_HIM(cem_img,sigma);
                sf_img = cem_img(:,:,1);
                threshold = graythresh(sf_img);
                old_map = imbinarize(sf_img,threshold);
                update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(TCIMF_LCMV_HOCDver(update_HIM,all_indexes(:, itr_num+1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf, indexes,pre_clsr_indexes));
                    cem_img = gaus_HIM(cem_img,sigma);
                    sf_img = cem_img(:,:,1);
                    threshold = graythresh(sf_img);
                    new_map = imbinarize(sf_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                end
                clear update_HIM
            case 5 % R-IRTS-IKTCIMF
                classifier_name = 'R-IRTS-IKTCIMF';
                TI = 0;
                indexes = this_class_idx(:,1)';
                if k==1
                    pre_clsr_indexes = [];
                else
                    pre_clsr_indexes = [];
                    rmv_class = group{k-1};
                    for q = 1:length(rmv_class)
                        pre_clsr_indexes = [pre_clsr_indexes;clsR_indexes_ccp{rmv_class(q),:}];
                    end
                    pre_clsr_indexes = pre_clsr_indexes';
                end
                group_tcimf = [1:class_no];
                group_tcimf(ordered_class(k)) = [];
                cem_img = abs(TCIMF_LCMV_HOCDver(update_HIM,all_indexes(:, 1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf, indexes,pre_clsr_indexes));
                cem_img = gaus_HIM(cem_img,sigma);
                sf_img = cem_img(:,:,1);
                threshold = graythresh(sf_img);
                old_map = imbinarize(sf_img,threshold);
                update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(TCIMF_LCMV_HOCDver(update_HIM,all_indexes(:, itr_num+1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf, indexes,pre_clsr_indexes));
                    cem_img = gaus_HIM(cem_img,sigma);
                    sf_img = cem_img(:,:,1);
                    threshold = graythresh(sf_img);
                    new_map = imbinarize(sf_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                end
                clear update_HIM
            case 6 % P?-IRTS-IKTCIMF
                classifier_name = 'Puper-IRTS-IKTCIMF';
                TI = 0;
                indexes = this_class_idx(:,1)';
                group_tcimf = [1:class_no];
                group_tcimf(ordered_class(k)) = [];
                cem_img = abs(TCIMF_LCMV_noRm(update_HIM,all_indexes(:, 1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf));
                cem_img = gaus_HIM(cem_img,sigma);
                sf_img = cem_img(:,:,1);
                threshold = graythresh(sf_img);
                old_map = imbinarize(sf_img,threshold);
                update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                itr_num = 1;
                while (TI < this_TIsetting) && (itr_num <= loop_limit)
                    % Normalize
                    [m2,n2,o2] = size(update_HIM);
                    update_HIM2d = reshape(update_HIM,m2*n2,o2);
                    [update_HIM2d_reg,M,N] = scale_func(update_HIM2d);
                    update_HIM = reshape(update_HIM2d_reg,m2,n2,o2);
                    
                    indexes = this_class_idx(:,itr_num+1)';
                    cem_img = abs(TCIMF_LCMV_noRm(update_HIM,all_indexes(:, itr_num+1),options,gt,nys_flag,para_set.n_sr,ordered_class,k,group_tcimf));
                    cem_img = gaus_HIM(cem_img,sigma);
                    sf_img = cem_img(:,:,1);
                    threshold = graythresh(sf_img);
                    new_map = imbinarize(sf_img,threshold);
                    saveTI(itr_num,class_th) = TI;
                    TI = Tanimoto_index(old_map, new_map);
                    old_map = new_map;
                    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(double(new_map),GT_class(:,:,class_th),class_p);
                    saveTI(itr_num,class_th) = TI;
                    savePA(itr_num,class_th) = Pa;
                    savePR(itr_num,class_th) = Pr;
                    disp([num2str(k) '-th Class' num2str(ordered_class(k)) ': ' num2str(itr_num) '; TI = ' num2str(TI) '; PA = ' num2str(Pa) '; PR = ' num2str(Pr)]);
                    itr_num = itr_num + 1;
                    clear threshold
                    update_HIM(:,:,end+1:end+size(cem_img,3)) = cem_img;
                end
                clear update_HIM
        end
        
        %% OTSU
        b_map = new_map;
        reg = reshape(b_map,m*n,1);
        clsR_indexes{k,1} = find(reg==1);
        clsR_indexes_ccp{ordered_class(k),1} = find(reg==1);
        if size(cem_img,3)==1
            SF_map(:,:,ordered_class(k)) = cem_img;
        else
            SF_map(:,:,ordered_class(k)) = cem_img(:,:,1);
        end
        result_otsu(:,:,ordered_class(k)) = b_map;
        %% OSP
        % Calculate class mean from classified data
        reg_x2d = HIM_2d;
        b_map2d = reshape(b_map,size(HIM_2d,1),1);
        if OSP_data == 1 || OSP_data == 3% Cls
            reg_x2d((b_map2d~=1),:) = [];
        elseif  OSP_data == 2% GT
            reg_x2d(gt~=class_th,:) = [];
        end
        if k~=class_no && OSP_data ~= 3
            disp(['Class' num2str(ordered_class(k+1)) ': remove class' num2str(group{k})]);
        end
        all_rinv_class(ordered_class(k)) = {reg_x2d};
        clsR_mean(ordered_class(k),:) = mean(reg_x2d,1);
        %% After obtaining Classified results and calculate the similarity between classes based on the classified results
        if OSP_data == 3 && k < class_no
            %% Calculate class similarity
            for i = 1:k
                disp(['Similarity between' num2str(ordered_class(k+1)) 'and' num2str(ordered_class(i))]);
                cs_sam(ordered_class(k+1),ordered_class(i)) = SAM(class_mean(ordered_class(k+1),:),clsR_mean(ordered_class(i),:));
                cs_ed(ordered_class(k+1),ordered_class(i)) = ED(class_mean(ordered_class(k+1),:),clsR_mean(ordered_class(i),:));
                cs_sid(ordered_class(k+1),ordered_class(i)) = SID(class_mean(ordered_class(k+1),:),clsR_mean(ordered_class(i),:));
            end
            %% Use threshold to determine the remove group
            this_class = ordered_class(k+1);
            reg_group = ordered_class(1:k)';
            switch CS_method
                case 1
                    cs_name = 'SAM';
                    class_thidx = find(cs_sam(this_class,:)>=CS_threshold);
                case 2
                    cs_name = 'SID';
                    class_thidx = find(cs_sid(this_class,:)>=CS_threshold);
                case 3
                    cs_name = 'ED';
                    class_thidx = find(cs_ed(this_class,:)>=CS_threshold);
            end
            reg_group(~ismember(reg_group,class_thidx)) = [];
            group(k,:) = {reg_group};
        end
        if classifier_type==2 || classifier_type==5% only R, no P?
            X_new = reshape(HIM_2d,m,n,o);
        else
            if k < class_no && ~isempty(group{k})
                for x = 1:length(group{k})
                    m_hat = mean(all_rinv_class{group{k}(x)},1);
                    U = [U m_hat'];%
                end
                % OSP
                U_psudoinv = inv(U'*U)*U';
                pUper = I-(U*U_psudoinv);
                % Update X
                reg_X = pUper*HIM_2d';
                X_new = reshape(reg_X',m,n,o);
            else
                X_new = reshape(HIM_2d,m,n,o);
            end
        end
        U = [];
    end
    if break_flag ==1
        disp(['TI setting (' num2str(this_TIsetting) ') is too high.'])
        break;
    end
    end_time = etime(clock,str_time);
    running_time = end_time;
    disp(['Total excution time: ' num2str(round(end_time,2)) ' secs']);
    %% Generate result by applying MAP to SFmap
    SFmap_2d = reshape(SF_map,m*n,class_no);
    [~,result_map] = max(SFmap_2d,[],2);
    result_map = reshape(result_map,m,n);
    %% Evaluate results
    % MAP result
    [PA,MS,PR,POA,PAA,POPR,PAPR,PPRnB,POPRnB,PAPRnB] = evaluation(result_map, GT);
    Summary = [PA PR];
    Summary = round(Summary,4)*100;
    Summary2 = [POA;PAA;POPR;PAPR];
    Summary2 = round(Summary2,4)*100;
    
    [Pa Pr Pr_b Poa Paa Popr Papr Popr_b Papr_b] = confusion(result_otsu,GT,class_p);
    Summary3 = [Pa Pr];
    Summary3 = round(Summary3,4)*100;
    Summary4 = [Poa;Paa;Popr;Papr];
    Summary4 = round(Summary4,4)*100;
    if loop == 1
        Map_resultPAPR = Summary;
        Map_resultOAPAPR = Summary2;
        Otsu_resultPAPR = Summary3;
        Otsu_resultOAPAPR = Summary4;
        result_map10 = result_map;
        result_otsu10(loop) = {result_otsu};
        running_time10 = running_time;
    else
        Map_resultPAPR = [Map_resultPAPR Summary];
        Map_resultOAPAPR = [Map_resultOAPAPR Summary2];
        Otsu_resultPAPR = [Otsu_resultPAPR Summary3];
        Otsu_resultOAPAPR = [Otsu_resultOAPAPR Summary4];
        result_map10(:,:,end+1) = result_map;
        result_otsu10(loop) = {result_otsu};
        running_time10 = [running_time10 running_time];
    end
    all_savePA(:,:,loop) = savePA;
    all_savePR(:,:,loop) = savePR;
    all_saveTI(:,:,loop) = saveTI;
    all_cs_sam(:,:,loop) = cs_sam;
    all_cs_sid(:,:,loop) = cs_sid;
    all_cs_ed(:,:,loop) = cs_ed;
    all_group(:,loop) = group;
end
% if break_flag ==0
if OSP_data == 1 || OSP_data == 3
    savename = [hsi_name num2str(train_saple_rate) '_HOCC' classifier_name '_OSPClsR_' num2str(extime) 'times'];
else
    savename = [hsi_name num2str(train_saple_rate) '_HOCC' classifier_name '_OSPGT_' num2str(extime) 'times'];
end
if CCP_type == 1
    savename = [savename '_CCPRINV'];
else
    savename = [savename '_CCPOSP'];
end
savename = [savename '_ksig' num2str(options.t) ];
PAotsu = Otsu_resultPAPR(:,1:2:end);
PRotsu = Otsu_resultPAPR(:,2:2:end);
PAmap = Map_resultPAPR(:,1:2:end);
PRmap = Map_resultPAPR(:,2:2:end);
% mean & std of 10 times
PAotsu_mean_std = [round(mean(PAotsu,2),2) round(std(PAotsu,0,2),2)];
PRotsu_mean_std = [round(mean(PRotsu,2),2) round(std(PRotsu,0,2),2)];
OAPAPRotsu_mean_std = [round(mean(Otsu_resultOAPAPR,2),2) round(std(Otsu_resultOAPAPR,0,2),2)];
PAmap_mean_std = [round(mean(PAmap,2),2) round(std(PAmap,0,2),2)];
PRmap_mean_std = [round(mean(PRmap,2),2) round(std(PRmap,0,2),2)];
OAPAPRmap_mean_std = [round(mean(Map_resultOAPAPR,2),2) round(std(Map_resultOAPAPR,0,2),2)];
time_mean_std = [round(mean(running_time10),2) round(std(running_time10),2)];
total_otsu1 = [PAotsu_mean_std PRotsu_mean_std];
total_otsu2 = [OAPAPRotsu_mean_std;time_mean_std];
total_map1 = [PAmap_mean_std PRmap_mean_std];
total_map2 = [OAPAPRmap_mean_std;time_mean_std];
if OSP_data == 3
    saveFolder = '.\Result_HOCDOSP_Cls\';
elseif OSP_data == 2
    saveFolder = '.\Result_HOCDOSP_GT\';
end
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder)
end
if TI_type == 1
    save([saveFolder savename '_' cs_name num2str(CS_threshold) '_TI'  num2str(TI_setting) '.mat'],'result_map10','result_otsu10','running_time10','Map_resultPAPR','Map_resultOAPAPR','Otsu_resultPAPR','Otsu_resultOAPAPR','PAotsu_mean_std','PRotsu_mean_std','OAPAPRotsu_mean_std','PAmap_mean_std','PRmap_mean_std','OAPAPRmap_mean_std','time_mean_std','total_otsu1','total_otsu2','total_map1','total_map2','SF_map','all_group','all_cs_sam','all_cs_sid','all_cs_ed','all_saveTI','TI_setting','all_savePA','all_savePR');
else
    save([saveFolder savename '_' cs_name num2str(CS_threshold) '_TIEachClass.mat'],'result_map10','result_otsu10','running_time10','Map_resultPAPR','Map_resultOAPAPR','Otsu_resultPAPR','Otsu_resultOAPAPR','PAotsu_mean_std','PRotsu_mean_std','OAPAPRotsu_mean_std','PAmap_mean_std','PRmap_mean_std','OAPAPRmap_mean_std','time_mean_std','total_otsu1','total_otsu2','total_map1','total_map2','SF_map','all_group','all_cs_sam','all_cs_sid','all_cs_ed','all_saveTI','TI_setting','all_savePA','all_savePR');
end
all_end_time = etime(clock,all_str_time);
disp(['Total ' num2str(extime) '-times excution time: ' num2str(fix(all_end_time)) ' secs']);
figure;
hold on
for i = 1:class_no
    subplot(2,8,i);
    imshow(SF_map(:,:,ordered_class(i)),[]);
    title([num2str(i) ': Class' num2str(ordered_class(i))])
end
sgtitle('HOCCOSPClsR')

reg = fix(max(all_saveTI) * 100) / 100;
reshapedArray = reshape(reg, [class_no, extime])';
min(reshapedArray)
disp(num2str(min(reshapedArray)))


