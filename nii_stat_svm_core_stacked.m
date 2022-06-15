function [acc, acc1, acc2, acc3, z_map1, z_map2, z_map3, acc_per_class, acc_per_class1, acc_per_class2, acc_per_class3, p, p1, p2, p3] = nii_stat_svm_core_stacked(data1, data2, class_labels, maxNsplits, normRowCol, verbose, islinear)
% Build multiple different models to classify multimodal data using linear
% support vector machines (one SVR model will be trained on data from modality
% 1, one SVR model will be trained on data from modality 2, one SVR model
% will be trained on data from both modalities, and one stacked model will
% generated a weighted 
%
% INPUTS:
% data1 (#observations x #features) -- data matrix (ROI or voxelwise) for
% modality 1
% data2 (#observations x #features) -- data matrix (ROI or voxelwise)for
% modality 2
% class_labels -- zeros and ones for observations in class 1 and 2
% maxNsplits -- number of splits in the cross-validation procedure
%               (recommended: 100 for voxelwise analysis, 500 for ROI analysis)
% normRowCol -- normalize none [0], rows [1, default], or columns [2]
% verbose -- if true, libsvm generates a lot of messages
% islinear -- should we use linear kernel for SVM (true by default, set to
%             false at your own risk)
% OUTPUTS:
% acc -- classification accuracy (proportion of correct out-of-sample
% assignments) for stacked model. Stacked model produces weighted average
% of models built only on data1 and data2 using logistic regression.
% acc1 -- classification accuracy (proportion of correct out-of-sample
% assignments) for model built using only data1.
% acc2 -- classification accuracy (proportion of correct out-of-sample
% assignments) for model built using only data2.
% acc3 -- classification accuracy (proportion of correct out-of-sample
% assignments) for model that pools data together (i.e., data1 and data2).
% z_map1 -- map of Z-scored feature weights (features being ROIs or voxels)
% for model built only using data1.
% z_map2 -- map of Z-scored feature weights (features being ROIs or voxels)
% for model built only using data2.
% z_map3 -- map of Z-scored feature weights (features being ROIs or voxels)
% for model built by pooling data1 and data2.
% acc_per_class -- accuracy for, separately, class 1 and class 2 (for
% stacked model)
% acc_per_class1 -- accuracy for, separately, class 1 and class 2 (for
% model built using only data1)
% acc_per_class2 -- accuracy for, separately, class 1 and class 2 (for
% model built only using data2)
% acc_per_class3 -- accuracy for, separately, class 1 and class 2 (for
% pooled data model)
% p -- p-value of classification accuracy, assuming binomial distribution
% (for stacked model)
% p1 -- p-value of classification accuracy, assuming binomial distribution
% (for model built with data1)
% p2 -- p-value of classification accuracy, assuming binomial distribution
% (for model built with data2)
% p3 -- p-value of classification accuracy, assuming binomial distribution
% (for pooled model)
%
% Original code from NiiStat, edited by A.T. to take in two datasets, build
% stacked model, and compare to the other kinds of models we can make. For
% comparisons, it's critical that each model is built using the same data
% which is why we build all possible models at the same time here
% (alex.teghipco@sc.edu)

% should we optimize C parameter for SVM? it's worth it, but takes time
optimize_C = true;

if ~exist('maxNsplits','var')
    maxNsplits = 100; 
end
if ~exist('normRowCol','var')
    normRowCol = 1; %none [0], rows [1], or columns [2]
end
if ~exist('verbose','var') %vebosity not specified
    verbose=false;
end
svmdir = [fileparts(which('nii_stat_svm_core.m'))  filesep 'libsvm' filesep];
if ~exist(svmdir,'file') 
    error('Unable to find utility scripts folder %s',svmdir);
end
if ~exist('islinear','var')
    islinear = true;
end
if ~islinear
    fprintf('Warning: please do not use loading map when conducting non-linear svm!');
end
addpath(svmdir); %make sure we can find the utility scripts


%for aalcat we may want to remove one hemisphere
%data(:,1:2:end)=[]; % Remove odd COLUMNS: left in AALCAT: analyze right
%data(:,2:2:end)=[]; % Remove even COLUMNS: right in AALCAT: analyze left
[data1, good_idx1]  = requireVarSub(data1);
[data2, good_idx2]  = requireVarSub(data2);
data3 = [data1 data2];
[data3, good_idx3]  = requireVarSub(data3);
%data = requireNonZeroSub(data); %CBF only

%normalize each predictor for range 0..1 to make magnitudes similar
%
%data  = normColSub(data);
if normRowCol ==  -2% rows [1], or columns [2], minus means exclude final column
    fprintf('Normalizing so each column has range 0..1 (last column excluded)\n');
    data1(:,1:end-1) = normColSub(data1(:,1:end-1));
    data2(:,1:end-1) = normColSub(data2(:,1:end-1));
    data3(:,1:end-1) = normColSub(data3(:,1:end-1));
elseif normRowCol ==  -1% rows [1], or columns [2], minus means exclude final column
    fprintf('Normalizing so each row has range 0..1 (last column excluded)\n');
    data1(:,1:end-1) = normRowSub(data1(:,1:end-1));
    data2(:,1:end-1) = normRowSub(data2(:,1:end-1));
    data3(:,1:end-1) = normRowSub(data3(:,1:end-1));
elseif normRowCol ==  1% rows [1], or columns [2]
    fprintf('Normalizing so each row has range 0..1\n');
    data1 = normRowSub(data1);
    data2 = normRowSub(data2);
    data3 = normRowSub(data3);
elseif normRowCol ==  2% rows [1], or columns [2]
    fprintf('Normalizing so each column has range 0..1\n');
    data1 = normColSub(data1);
    data2 = normColSub(data2);
    data3 = normColSub(data3);
end
%data  = normRowSub(data);
%binarize class_label: either 0 or 1
if (min(class_labels) == max(class_labels))
    error('No variability in class labels (final column of file)');
end
if ~exist('thresholdHi','var') %if Excel file not specified, have user select one
    fprintf('Class values from %g..%g (median %g)\n',min(class_labels(:)), max(class_labels(:)), median(class_labels(:)));
    mdn = median(class_labels(:));
    if mdn > min(class_labels(:))
        thresholdHi = mdn;
        %thresholdLo = thresholdHi-eps;
        thresholdLo = max(class_labels(class_labels < mdn));
    else
        thresholdLo = mdn;
        thresholdHi = min(class_labels(class_labels > mdn));
    end
    %answer = inputdlg({'Class1 no more than','Class2 no less than'}, sprintf('Threshold (input range %g..%g)',min(class_labels),max(class_labels)), 1,{num2str(min(class_labels)),num2str(max(class_labels))});
    %thresholdLo = str2double (cell2mat(answer(1)));
    %thresholdHi = str2double (cell2mat(answer(2)));
end
%fprintf('Class labels range from %g to %g, values of %g or less will be group0, values of %g or more will be group1\n', min(class_labels), max(class_labels), thresholdLo, thresholdHi);
%fprintf('Processing the command line: \n');
%fprintf(' %s (''%s'', %d, %g, %g, %g, %g);\n', mfilename, xlsname, normRowCol, thresholdLo, thresholdHi, verbose, islinear);

[class_labels , data1] = binarySub(class_labels, data1, thresholdLo, thresholdHi);
[~ , data2] = binarySub(class_labels, data2, thresholdLo, thresholdHi);
[~ , data3] = binarySub(class_labels, data3, thresholdLo, thresholdHi);

if islinear 
    cmd = '-t 0';
else
    cmd = '-t 2';
end
    
class1_idx = find (class_labels == 1)';
class0_idx = find (class_labels == 0)';
n0 = length (class0_idx); 
n1 = length (class1_idx); % # examples per class
N = n0 + n1;
min_n = min (n0, n1);
if (n1 < 2) || (n0 < 2)
    fprintf('Each group must have at least 2 observations: please use a different class threshold\n');
    z_map = []; acc = []; acc_per_class = []; p = [];
    return;
end
data1 = data1'; 
data2 = data2';
data3 = data3';

if optimize_C
    N_splits = 20; % quick and dirty split-half cross-validation to select C
    C_list = [0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5];
    for g = 1:N_splits
        shuffle = randperm (n0);
        idx01 = class0_idx (sort(shuffle (1:floor(min_n/2))));
        idx02 = class0_idx (sort(shuffle (floor(min_n/2)+1:min_n)));
        shuffle = randperm (n1);
        idx11 = class1_idx (sort(shuffle (1:floor(min_n/2))));
        idx12 = class1_idx (sort(shuffle (floor(min_n/2)+1:min_n)));
        
        data1_1 = data1 (:, [idx01 idx11]); data1_2 = data1 (:, [idx02 idx12]);
        data2_1 = data2 (:, [idx01 idx11]); data2_2 = data2 (:, [idx02 idx12]);
        data3_1 = data3 (:, [idx01 idx11]); data3_2 = data3 (:, [idx02 idx12]);
        class_labels1 = class_labels ([idx01 idx11]); class_labels2 = class_labels ([idx02 idx12]);
        for C_idx = 1:length(C_list)
            str = sprintf ('''%s -c %g''', cmd, C_list(C_idx));
            [out, subSVM1_1] = evalc (['svmtrain (class_labels1, data1_1'', ' str ');']);
            [out, subSVM1_2] = evalc (['svmtrain (class_labels2, data1_2'', ' str ');']);
            ww1_1 = subSVM1_1.sv_coef' * subSVM1_1.SVs;
            ww1_2 = subSVM1_2.sv_coef' * subSVM1_2.SVs;
            temp = corrcoef (ww1_1, ww1_2);
            repr1(C_idx, g) = temp (1, 2);
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels2, data1_2'', subSVM1_1)');
            acc1_1 = temp(1)/100;
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels1, data1_1'', subSVM1_2)');
            acc1_2 = temp(1)/100;
            acc1(C_idx, g) = mean ([acc1_1 acc1_2]);

            [out, subSVM2_1] = evalc (['svmtrain (class_labels1, data2_1'', ' str ');']);
            [out, subSVM2_2] = evalc (['svmtrain (class_labels2, data2_2'', ' str ');']);
            ww2_1 = subSVM2_1.sv_coef' * subSVM2_1.SVs;
            ww2_2 = subSVM2_2.sv_coef' * subSVM2_2.SVs;
            temp = corrcoef (ww2_1, ww2_2);
            repr2(C_idx, g) = temp (1, 2);
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels2, data2_2'', subSVM2_1)');
            acc2_1 = temp(1)/100;
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels1, data2_1'', subSVM2_2)');
            acc2_2 = temp(1)/100;
            acc2(C_idx, g) = mean ([acc2_1 acc2_2]);

            [out, subSVM3_1] = evalc (['svmtrain (class_labels1, data3_1'', ' str ');']);
            [out, subSVM3_2] = evalc (['svmtrain (class_labels2, data3_2'', ' str ');']);
            ww3_1 = subSVM3_1.sv_coef' * subSVM3_1.SVs;
            ww3_2 = subSVM3_2.sv_coef' * subSVM3_2.SVs;
            temp = corrcoef (ww3_1, ww3_2);
            repr3(C_idx, g) = temp (1, 2);
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels2, data3_2'', subSVM3_1)');
            acc3_1 = temp(1)/100;
            [~, ~, temp, ~] = evalc ('svmpredict (class_labels1, data3_1'', subSVM3_2)');
            acc3_2 = temp(1)/100;
            acc3(C_idx, g) = mean ([acc3_1 acc3_2]);
        end  
    end

    cost1 = ((1+acc1)/2).^2 + repr1.^2;    
    [~, optC_idx] = max (mean (cost1, 2));
    C1 = C_list (optC_idx);
    fprintf ('Optimized value of C for data1: %g\n', C1);

    cost2 = ((1+acc2)/2).^2 + repr2.^2;    
    [~, optC_idx] = max (mean (cost2, 2));
    C2 = C_list (optC_idx);
    fprintf ('Optimized value of C for data2: %g\n', C2);

    cost3 = ((1+acc3)/2).^2 + repr3.^2;    
    [~, optC_idx] = max (mean (cost3, 2));
    C3 = C_list (optC_idx);
    fprintf ('Optimized value of C for pooled data: %g\n', C3);
else
    C1 = 0.01;
    C2 = 0.01;
    C3 = 0.01;
    fprintf ('No optimization of C; using default of %g\n', C1);
end
cmd1 = sprintf ('%s -c %g', cmd, C1);
cmd2 = sprintf ('%s -c %g', cmd, C2);
cmd3 = sprintf ('%s -c %g', cmd, C3);
clear acc1 acc2 acc3 repr1 repr2 repr3;

n_train = min_n - 1; % number of training examples per class
n_test0 = n0 - n_train; % number of test examples in class 1
n_test1 = n1 - n_train; % ... and in class 2
% maximum number of training-test splits
warning ('OFF', 'MATLAB:nchoosek:LargeCoefficient');
theoretical_max_n_splits = nchoosek (n1, n_test1) * nchoosek (n0, n_test0);
n_splits = min (theoretical_max_n_splits, maxNsplits);
correct = zeros (size (class_labels));
correct1 = zeros (size (class_labels));
correct2 = zeros (size (class_labels));
correct3 = zeros (size (class_labels));
ntrials = zeros (size (class_labels)); % number of times each example was tested
prev_splits = zeros (1, 2*n_train);
curr_split = zeros (1, 2*n_train); % reset at first iteration
for split = 1:n_splits
    % make sure a newly created split is unique
    while ~isempty (find (ismember (prev_splits, curr_split, 'rows'))) %#ok<EFIND>
        shuffle = randperm (n1);
        train_idx1 = class1_idx(sort (shuffle (1:n_train)));
        test_idx1 = class1_idx(sort (shuffle (n_train+1:n1)));
        shuffle = randperm (n0);
        train_idx0 = class0_idx(sort (shuffle (1:n_train)));
        test_idx0 = class0_idx(sort (shuffle (n_train+1:n0)));
        curr_split = [train_idx1 train_idx0];
    end
    prev_splits (split, :) = curr_split;
end
for split = 1:n_splits
    used(split,:) = setdiff([1:56],prev_splits (split, :));
end

    test_idx = [test_idx1 test_idx0];
    train_idx = [train_idx1 train_idx0];
    test_data1 = data1 (:, test_idx);
    test_data2 = data2 (:, test_idx);
    test_data3 = data3 (:, test_idx);
    test_labels = class_labels (test_idx)';
    train_data1 = data1 (:, train_idx);
    train_data2 = data2 (:, train_idx);
    train_data3 = data3 (:, train_idx);
    train_labels = class_labels (train_idx)';  

   	if verbose
        model1 = svmtrain (train_labels', train_data1', cmd1);
        model2 = svmtrain (train_labels', train_data2', cmd2);
        model3 = svmtrain (train_labels', train_data2', cmd3);
        [svtclass1, ~, svtprob1] = svmpredict (train_labels', train_data1', model1);
        [svtclass2, ~, svtprob2] = svmpredict (train_labels', train_data2', model2);
        [assignments1, ~, prob1] = svmpredict (test_labels', test_data1', model1);
        [assignments2, ~, prob2] = svmpredict (test_labels', test_data2', model2);
        [assignments3, ~, ~] = svmpredict (test_labels', test_data3', model3);
    else %if verbose else silent
        [~, model1] = evalc ('svmtrain (train_labels'', train_data1'', cmd1)'); %-t 0 = linear
        [~, model2] = evalc ('svmtrain (train_labels'', train_data2'', cmd2)'); %-t 0 = linear
        [~, model3] = evalc ('svmtrain (train_labels'', train_data3'', cmd3)'); %-t 0 = linear
        [~, svtclass1, ~, svtprob1] = evalc ('svmpredict (train_labels'', train_data1'', model1)');
        [~, svtclass2, ~, svtprob2] = evalc ('svmpredict (train_labels'', train_data2'', model2)');
        [~, assignments1, ~, prob1] = evalc ('svmpredict (test_labels'', test_data1'', model1)');
        [~, assignments2, ~, prob2] = evalc ('svmpredict (test_labels'', test_data2'', model2)');
        [~, assignments3, ~, ~] = evalc ('svmpredict (test_labels'', test_data3'', model3)');
    end %if verbose...
    
    % now build a model to predict training data using multinomial logistic
    % regression
    mdl = fitclinear([svtprob1 svtprob2],train_labels,'Learner','logistic');
    assignments = predict(mdl,[prob1 prob2]);

    ntrials (test_idx) = ntrials (test_idx) + 1;

    map1 (split, :) = model1.sv_coef' * model1.SVs; %#ok<AGROW>
    correct_idx = test_idx (find (test_labels == assignments1'));
    correct1 (correct_idx) = correct1 (correct_idx) + 1;    

    map2 (split, :) = model2.sv_coef' * model2.SVs; %#ok<AGROW>
    correct_idx = test_idx (find (test_labels == assignments2'));
    correct2 (correct_idx) = correct2 (correct_idx) + 1; 

    map3 (split, :) = model3.sv_coef' * model3.SVs; %#ok<AGROW>
    correct_idx = test_idx (find (test_labels == assignments3'));
    correct3 (correct_idx) = correct3 (correct_idx) + 1;

    correct_idx = test_idx (find (test_labels == assignments'));
    correct (correct_idx) = correct (correct_idx) + 1; 
end %for each split  

% mean_map = mean (map, 1);
% z_map = mean_map ./ std (mean_map);
% Z-scoring of maps: a version of delete-d jackknife
d = abs (n0 - n1) + 2; 
t_map1 = mean (map1, 1) ./ (std (map1, 1, 1) * sqrt(N/d-1));
t_map2 = mean (map2, 1) ./ (std (map2, 1, 1) * sqrt(N/d-1));
t_map3 = mean (map3, 1) ./ (std (map3, 1, 1) * sqrt(N/d-1));
z_map1 = zeros (size (t_map1));
z_map1 (:) = nan;
z_map1 (~isnan (t_map1)) = spm_t2z (t_map1(~isnan (t_map1)), length(class_labels) - 1);
z_map2 = zeros (size (t_map2));
z_map2 (:) = nan;
z_map2 (~isnan (t_map2)) = spm_t2z (t_map2(~isnan (t_map2)), length(class_labels) - 1);
z_map3 = zeros (size (t_map3));
z_map3 (:) = nan;
z_map3 (~isnan (t_map3)) = spm_t2z (t_map3(~isnan (t_map3)), length(class_labels) - 1);

if exist('good_idx1','var')  %insert NaN for unused features
    z_mapOK = zeros(size(data1, 1), 1);
    z_mapOK(:) = nan;
    z_mapOK(good_idx1) = z_map1;
    z_map1 = z_mapOK;
end
if exist('good_idx2','var')  %insert NaN for unused features
    z_mapOK = zeros(size(data2, 1), 1);
    z_mapOK(:) = nan;
    z_mapOK(good_idx2) = z_map2;
    z_map2 = z_mapOK;
end
if exist('good_idx3','var')  %insert NaN for unused features
    z_mapOK = zeros(size(data3, 1), 1);
    z_mapOK(:) = nan;
    z_mapOK(good_idx3) = z_map3;
    z_map3 = z_mapOK;
end
temp1 = correct1 ./ ntrials;
acc1 = mean (temp1 (find (~isnan (temp1)))); %#ok<*FNDSB>
acc_per_class1(1) = mean (temp1 (intersect (class1_idx, find (~isnan (temp1)))));
acc_per_class1(2) = mean (temp1 (intersect (class0_idx, find (~isnan (temp1)))));
temp2 = correct2 ./ ntrials;
acc2 = mean (temp2 (find (~isnan (temp2)))); %#ok<*FNDSB>
acc_per_class2(1) = mean (temp2 (intersect (class1_idx, find (~isnan (temp2)))));
acc_per_class2(2) = mean (temp2 (intersect (class0_idx, find (~isnan (temp2)))));
temp3 = correct3 ./ ntrials;
acc3 = mean (temp3 (find (~isnan (temp3)))); %#ok<*FNDSB>
acc_per_class3(1) = mean (temp3 (intersect (class1_idx, find (~isnan (temp3)))));
acc_per_class3(2) = mean (temp3 (intersect (class0_idx, find (~isnan (temp3)))));
temp = correct ./ ntrials;
acc = mean (temp (find (~isnan (temp)))); %#ok<*FNDSB>
acc_per_class(1) = mean (temp (intersect (class1_idx, find (~isnan (temp)))));
acc_per_class(2) = mean (temp (intersect (class0_idx, find (~isnan (temp)))));

%report results
probb = max(n0,n1)/(n0+n1);
p = bipSub(acc*n_splits, n_splits, probb);
p1 = bipSub(acc1*n_splits, n_splits, probb);
p2 = bipSub(acc2*n_splits, n_splits, probb);
p3 = bipSub(acc3*n_splits, n_splits, probb);

fprintf('Observed %d in group0 and %d in group1 (prob = %g%%) with %d predictors\n', n0, n1,max(n0,n1)/(n0+n1), size(data1,1)+size(data2,1));
fprintf('For stacked model: BinomialProbality nHits= %g, nTotal= %g, IncidenceOfCommonClass= %g, p< %g\n', acc*n_splits, n_splits,probb, p);
fprintf('For stacked model: Overall Accuracy %g (%g for group0, %g for group1)\n', acc, acc_per_class(2),acc_per_class(1));
fprintf('For stacked model: Thresh0\t%g\tThresh1\t%g\tn0\t%d\tn1\t%d\tProb\t%g\tAcc\t%g\tAcc0\t%g\tAcc1\t%g\tp<\t%g\n',...
    thresholdLo,thresholdHi,n0,n1,probb,acc,acc_per_class(2),acc_per_class(1),p);
fprintf('For pooled model: BinomialProbality nHits= %g, nTotal= %g, IncidenceOfCommonClass= %g, p< %g\n', acc3*n_splits, n_splits,probb, p3);
fprintf('For pooled model: Overall Accuracy %g (%g for group0, %g for group1)\n', acc3, acc_per_class3(2),acc_per_class3(1));
fprintf('For pooled model: Thresh0\t%g\tThresh1\t%g\tn0\t%d\tn1\t%d\tProb\t%g\tAcc\t%g\tAcc0\t%g\tAcc1\t%g\tp<\t%g\n',...
    thresholdLo,thresholdHi,n0,n1,probb,acc3,acc_per_class3(2),acc_per_class3(1),p3);

% ----- MAIN FUNCTION ENDS

function prob = bipSub(x,n,p)
%report extreme tail: chance for equal or more extreme values
if x > (p*n)
    h = n - x;
    px = 1-p;
else
   h = x; 
   px = p;
end
prob = binocdfSub(h, n, px);
%end

function cdf = binocdfSub (x, n, p)
% probability of at least x or more correct in n attempts, where probability of correct is p
%  x : number of successes, in range 0..n (for accuracy of 0..100%)
%  n : number of attempts (integer)
%  p : chance probability of success
% http://en.wikipedia.org/wiki/Binomial_distribution
%Normal Approximation when values will overflow factorial version
% http://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation
% http://code.metager.de/source/xref/gnu/octave/scripts/statistics/distributions/binocdf.m
if  ((n*p) > 9) && ((n * (1-p)) > 9) %use approximation
    k = (x >= 0) & (x < n) & (n == fix (n)) & (p >= 0) & (p <= 1);
    tmp = floor (x(k));
    cdf = betainc (1 - p, n - tmp, tmp + 1);
    return
end
cdf = 0.0;
for i = 0:x
        cdf = cdf + nchoosek(n, i)*(power(p,i) )*(power(1.0-p,(n-i) ) );
end

% function p = bipSub(nHits, nTotal, prob)
% % probability of at least nHits correct classifications in nTotal attempts
% % nHits is in range 0..n (for accuracy of 0..100%)
% % http://en.wikipedia.org/wiki/Binomial_distribution
% %Approximation suitable for large values that would overflow factorial version
% % http://www.dummies.com/how-to/content/how-to-find-the-normal-approximation-to-the-binomi.html
% np = nTotal * prob;
% n_not_p = nTotal * (1-prob);
% if (np > 10) && (n_not_p > 10)
%     var = sqrt(np * (1.0 - prob));
%     z = (nHits-np)/var;
%     p = 1 - spm_Ncdf(z);
%     return
% end
% if nHits > (prob*nTotal)
%     nHitsSmallTail = nTotal - nHits;
% else
%    nHitsSmallTail = nHits; 
% end
% p = 0.0;
% for lHitCnt = 0:nHitsSmallTail
%     p = p + nchoosek(nTotal, lHitCnt)*(power(prob,lHitCnt) )*(power(1.0-prob,(nTotal-lHitCnt) ) );
% end
% %end bipSub()

% function x = normColSub(y)
% %normalize each column for range 0..1
% x = y';
% mn = min(x); %minimum
% rng = max(x) - mn; %range
% rng(rng == 0) = 1; %avoid divide by zero
% rng = 1 ./ rng; %reciprocal: muls faster than divs
% for i = 1 : numel(mn)
%     x(:,i)=(x(:,i)-mn(i)) * rng(i);
% end
% x = x';
% %normColSub

function x = normColSub(x)
%normalize each column for range 0..1
% x = [1 4 3 0; 2 6 2 5; 3 10 2 2] -> x = [0 0 1 0; 0.5 0.333 0 1; 1 1 0 0.4]
if size(x,1) < 2 %must have at least 2 rows
    fprintf('Error: normalizing columns requires multiple rows\n');
    return
end
if min(max(x,[],1)-min(x,[],1)) == 0
    fprintf('Error: unable to normalize columns: some have no variability\n');
    return;
end
x = bsxfun(@minus,x,min(x,[],1)); %translate so minimum = 0
x = bsxfun(@rdivide,x,max(x,[],1)); %scale so range is 1
%end normColSub()

function x = normRowSub(x)
%normalize each column for range 0..1
% x = [1 4 3 0; 2 6 2 5; 3 10 2 2] -> x = [0.25 1 0.75 0; 0 1 0 0.75; 0.125 1 0 0]
if size(x,2) < 2 %must have at least 2 rows
    fprintf('Error: normalizing rows requires multiple columns\n');
    return
end
if min(max(x,[],2)-min(x,[],2)) == 0
    fprintf('Error: unable to normalize rows: some have no variability\n');
    return;
end
x = bsxfun(@minus,x,min(x,[],2)); %translate so minimum = 0
x = bsxfun(@rdivide,x,max(x,[],2)); %scale so range is 1
%end normRowSub()

function num = tabreadSub(tabname)
%read cells from tab based array. 
fid = fopen(tabname);
num = [];
row = 0;
while(1) 
	datline = fgetl(fid); % Get second row (first row of data)
	%if (length(datline)==1), break; end
    if(datline==-1), break; end %end of file
    if datline(1)=='#', continue; end; %skip lines that begin with # (comments)
    tabLocs= strfind(datline,char(9)); %findstr(char(9),datline); % find the tabs
    row = row + 1;
    if (row < 2) , continue; end; %skip first row 
    if (tabLocs < 1), continue; end; %skip first column
    dat=textscan(datline,'%s',(length(tabLocs)+1),'delimiter','\t');
    for col = 2: size(dat{1},1) %excel does not put tabs for empty cells (tabN+1)
    	num(row-1, col-1) = str2double(dat{1}{col}); %#ok<AGROW>
    end
end %while: for whole file
fclose(fid);
%end tabreadSub()

% function [good_dat, good_idx] = requireNonZeroSub (dat)
% %remove columns with zeros
% good_idx=[];
% for col = 1:size(dat,2)
%     if sum(dat(:,col) == 0) > 0
%        %fprintf('rejecting column %d (non-numeric data')\n',col) %
%     elseif min(dat(:,col)) ~= max(dat(:,col))
%         good_idx = [good_idx, col];  %#ok<AGROW>
%     end
% end %for col: each column
% if numel(good_idx) ~= size(dat,2)
%     fprintf('Some predictors have zeros (analyzing %d of %d predictors)\n',numel(good_idx), size(dat,2));
% end
% good_dat = dat(:,good_idx);
% %end requireNonZeroSub()

function [good_dat, good_idx] = requireVarSub (dat)
good_idx=[];
for col = 1:size(dat,2)
    if sum(isnan(dat(:,col))) > 0
       %fprintf('rejecting column %d (non-numeric data')\n',col) %
    elseif min(dat(:,col)) ~= max(dat(:,col))
        good_idx = [good_idx, col];  %#ok<AGROW>
    end
end %for col: each column
if sum(isnan(dat(:))) > 0
    fprintf('Some predictors have non-numeric values (e.g. not-a-number)\n');
end
if numel(good_idx) ~= size(dat,2)
    fprintf('Some predictors have no variability (analyzing %d of %d predictors)\n',numel(good_idx), size(dat,2));
end
good_dat = dat(:,good_idx);
%end requireVarSub()

%function dataBin = binarySub(data, threshold)
%dataBin = zeros(size(data));
%dataBin(data >= threshold) = 1;
%end binarySub

function [classBin,data] = binarySub(classIn, data, thresholdLo, thresholdHi)
%rows where classIn is <= thresholdLo are assigned group 0 in classBin
%rows where classIn is >= thresholdHi are assigned group 1 in classBin
%rows where classIn is between thresholdLo and ThresholdHi are deleted
classBin = zeros(numel(classIn),1);
classBin(classIn > thresholdLo) = nan;
classBin(classIn >= thresholdHi) = 1;
bad = isnan(classBin);
data(bad,:) = [];
classBin(bad)=[];
%end binarySub

