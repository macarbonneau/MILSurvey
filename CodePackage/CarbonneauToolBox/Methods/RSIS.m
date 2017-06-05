function out = RSIS(D,operation,model,opt)


switch operation
    case 'train'
        
        bestOpt = validationRSIS(D,opt);
        model = trainRSIS(D, bestOpt);
        
        out = model;
        
    case 'test'
        
        % classify update labels model
        out = classifyTestDataset(model,D);
        
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
end

end


function out = validationRSIS(D,opt)

nFolds = 5;
out = opt;
rng('shuffle')
expNum = randi(100000000);


% create bag list for folds
BagPerFoldList = divideBagsInFolds(nFolds,D);
[cfgList] = createConfigList(opt);
res = zeros(size(cfgList,1),1);

for fold = 1:nFolds
    
    
    %% create training and test datasets
    [TRD, TED] = getTrainingAndTestDatasets(fold,nFolds,BagPerFoldList,D);
    
    for c = 1:size(cfgList,1)
        tic
        o = opt;
        o.NDSS = cfgList(c,1);
        o.nK = cfgList(c,2);
        o.T = cfgList(c,3);
        
        models = trainRSIS(TRD,o);
        % test
        r = classifyTestDataset(models,TED);
        tmp = getClassifierPerfomance(r.PLB,TED.YB,r.SCB);
        disp([num2str(o.NDSS) ' ' num2str(o.nK) ' ' num2str(o.T) ' AUC:' num2str(tmp.AUC)])
        
        fn = ['temp/RSIS-CV-' num2str(expNum) '-' num2str(c)];
        dlmwrite(fn,tmp.AUC);
        toc
    end
    
    % compile results
    for  c = 1:size(cfgList,1)
        fn = ['temp/RSIS-CV-' num2str(expNum) '-' num2str(c)];
        tmp = dlmread(fn);
        res(c) = res(c)+tmp;
    end
        
end

% get the best performing parameters
[~,ind] = max(res);
c = ind(1);

 out.NDSS = cfgList(c,1);
 out.nK = cfgList(c,2);
 out.T = cfgList(c,3);

fprintf('\n')
disp(['the best config for RSIS is : NDSS =' num2str(out.NDSS) ...
    ' and nK =' num2str(out.nK) ' and T: ' num2str(out.T)])

end


function out = trainRSIS(D, opt)


nInit = 10;
pop = [];
%tic
%disp('Sub set construction')
for in = 1:nInit
    
    temp = getDatasetSelectedIndex(D,opt.ES/nInit,opt.nK,opt.nRSS,...
        opt.NDSS,opt.T);
    pop = [pop;temp];
    %fprintf('.')
end

nPI = sum(D.YR);
IOnce = sum(pop)>0;
nIOnce = sum(IOnce&D.YR');
percW = sum(pop)/size(pop,1);
percW = percW(D.YR==1);
percI = sum(pop)/size(pop,1);
percI = percI(IOnce&D.YR');


disp('=====================================================')
disp('= INENTIFICATION REPORT =')
disp(['Total Witness : ' num2str(nPI)])
disp(['Witness identified once : ' num2str(nIOnce)])
disp(['Witness mean : ' num2str(mean(percW))])
disp(['Correctly identified Witness mean: ' num2str(mean(percI))])
disp('=====================================================')

%toc
% get models trained with the selected instances
out = getSVMModels(pop,D,opt);
end

function selection = getDatasetSelectedIndex(D,nSel,nK,nRS,nDSS,T)


% D is a MILdataset object

% nK = 20;    % number of clusters in k-means
% nRS = 2000;   % number of random subspaces
% nDSS = 30;   % number of dimension per subspace
% T = .01;     % temperature for softmax selection

res = zeros(nRS,size(D.X,1));

nDSS = ceil(size(D.X,2)*nDSS);

%% Cluster in different subspaces
for s = 1:nRS
    
    % Randomly select the dimension for the subspace
    ind = randperm(size(D.X,2));
    ind = ind(1:nDSS);
    R = false(size(D.X,2), 1);
    R(ind) = true;
    
    % Cluster data in the subspace
    [C, A] = vl_kmeans(D.X(:,R)', nK);
    C = C';
    
    % compute proporition of positive bag per cluster
    pC = zeros(size(C,1),1);
    ctnC = zeros(size(C,1),1);
    
    for i = 1:length(A)
        ctnC(A(i)) = ctnC(A(i))+1;
        pC(A(i)) = pC(A(i))+D.Y(i);
    end
    pC = pC./ctnC;
    res(s,:) = pC(A)';
    
end


score = mean(res,1);
probX = score*0;

%% create softmax probability vector
bagList = unique(D.XtB);
for i = 1:length(bagList)
    
    % get denuminator
    idb = (D.XtB == bagList(i));
    denum = sum(exp(score(idb)/T));
    
    % get index of elements in the bag
    idi = 1:length(D.XtB);
    idi = idi(idb);
    
    for j = 1:length(idi)
        probX(idi(j)) = exp(score(idi(j))/T)/denum;
    end
    
end


%% Do a report
% probW = [];
% sL = [];
% probI = [];
% 
% for b = 1:length(D.YB)
% if D.YB(b) == 1
%    PB = probX(D.XtB==D.B(b));
%    LB = D.YR(D.XtB==D.B(b));
%    [~,i] = max(PB);
%    sL = [sL; LB(i)];
%    probW = [probW;PB(LB==1)];
%    probI = [probI;PB(i)];
% end
%     
% end
% 
% disp('=====================================================')
% disp('= PROBABILITY REPORT =')
% disp(['Number of Witness : ' num2str(length(probW))])
% disp(['mean prob Witness : ' num2str(mean(probW))])
% disp(['Witness identified once : ' num2str(sum(sL))])
% disp(['identify instance mean: ' num2str(mean(probI))])
% disp('=====================================================')


%% sample individual from softmax vector

selection = false(nSel,size(D.X,1));
for j = 1:nSel
    
    % select one instance per bag
    for i = 1:length(bagList)
        
        % get probabilities of instance of the bag
        idb = (D.XtB == bagList(i));
        p = probX(idb);
        
        % get index of elements in the bag
        idi = 1:length(D.XtB);
        idi = idi(idb);
        
        % get index of the selected instance
        t = rand(1);
        cumm = 0;
        for k = 1:length(idi)
            cumm = cumm + p(k);
            if t < cumm
                break;
            end
        end
        
        selection(j,idi(k)) = true;
        
    end
end

%% sample more instance from the negative bags


for j = 1:nSel
    
    % count the number of positive examples
    ip = selection(j,:) == 1 & logical(D.Y');
    np = sum(ip);
    
    % count the number of negative examples
    in = selection(j,:) == 1 & ~logical(D.Y');
    nn = sum(in);
    
    % get index of non-selected negative examples
    insn = selection(j,:) == 0 & ~logical(D.Y');
    temp = 1:size(selection,2);
    insn = temp(insn);
    
    temp = randperm(length(insn));
    insn = insn(temp);
    
    % if possible add enough negative sample to get desired ratio
    na = np*3-nn;
    if na > length(insn)
       na = length(insn);
    end
    selection(j,insn(1:na)) = true;
    
    
    % count the number of positive examples
    ip = selection(j,:) == 1 & logical(D.Y');
    np2 = sum(ip);
    
    % count the number of negative examples
    in = selection(j,:) == 1 & ~logical(D.Y');
    nn2 = sum(in);
    
    if nn2 > 3*np2
       disp(['WARNING: SVM class imabalance: ' num2str(np2/nn2)]) 
    end
    
end



end

function models = getSVMModels(pop,D,opt)

models = cell(size(pop,1),1);

% get a model for each individual
for ind = 1:size(pop,1)
    
    genes = pop(ind,:)';
    
    % create positive dataset
    Xpos = D.X(genes & D.Y==1,:);
    
    % create negative dataset
    Xneg = D.X(genes & D.Y==0,:);
    
    %% train classifier
    X = [Xpos;Xneg];
    Y = [ones(size(Xpos,1),1);zeros(size(Xneg,1),1)];
    opt.classImbalance = true;
    models{ind} = trainSVM_CV(X,Y,opt);
    
end


end

function out = classifyTestDataset(models,D)

insLabels = zeros(size(models,1),length(D.Y));


for m = 1:size(models,1)
    
    % test classifier on validation data
    [temp, ~, ~] = svmpredict(D.Y,D.X, models{m});
    insLabels(m,:) = temp';
end

% combine with average
out.SC = mean(insLabels)';
out.PL = double(out.SC > 0.5);

[out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);

end

function [cfgList] = createConfigList(opt)

cfgList = [];

for i1 = 1:length(opt.NDSS)
    for i2 = 1:length(opt.nK)
        for i3 = 1:length(opt.T)
            tmp = [opt.NDSS(i1) opt.nK(i2) opt.T(i3)];
            cfgList = [cfgList;tmp];
        end
    end
end
end

