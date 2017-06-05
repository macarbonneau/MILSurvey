
function out =  myMILES(D,operation,model)


switch operation
    case 'train'
        
        % get best configuration
        [C, gamma] = validationMILES(D);
        
        % convert data set to the pr format
        D = convertDatasetForMILToolbox(D);
        
        % get the model
        out = miles(D,C,'r',gamma);
        
    case 'test'
        
        % classify instances and bags
        out = classifyMILES(D,model);
         
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;      
end

end


function [C, gamma] = validationMILES(D)

% parameters
C = [0.01 0.1 1 10 100];

gamma = [0.5 1 2 4 8 16 32 64 128];

nFolds = 5;
tic
% matrix for saving results
AUC = zeros(length(C),length(gamma),nFolds);
Acc = AUC;

BagPerFoldList = divideBagsInFolds(nFolds,D);

for f = 1:nFolds
    
    statusBar(f,nFolds)
    [TRD, TED] = getTrainingAndTestDatasets(f,nFolds,BagPerFoldList,D);
    
    % convert data set
    disp(['Data set Conversion fold ' num2str(f)])
    tic
    TRD = convertDatasetForMILToolbox(TRD);
    TED = convertDatasetForMILToolbox(TED);
    disp(['Done'])
    toc
    
    parfor c = 1:length(C)
        
        AUCtmp = zeros(1,length(gamma));
        
        for g = 1:length(gamma)
            
            % train a model
            model = miles(TRD,C(c),'r',gamma(g));
            
            % test the model
            % get performance
            AUCtmp(g) = dd_auc(TED*model*milroc);
            disp(['C = ' num2str(C(c)) ' gamma = ' num2str(gamma(g))...
                '  AUC = ' num2str(AUCtmp(g))]);
            
        end
        
        fn = ['temp/tmpVal' num2str(C(c))];
        dlmwrite(fn,AUCtmp);
        
    end
    
    for c = 1:length(C)
        fn = ['temp/tmpVal' num2str(C(c))];
        tmp = dlmread(fn);
        AUC(c,:,f) = tmp;
        
    end
end

AUC = mean(AUC,3);

% get the best performing parameters
[~,ind] = max(AUC(:));
[c,g] = ind2sub(size(AUC),ind);

disp(['the best config for MILES is : C=' num2str(C(c)) ' and g=' num2str(gamma(g))])

C = C(c);
gamma = gamma(g);
toc
end


function out = classifyMILES(D,model)


%% BAG Classification

% convert data set to the pr format
a = convertDatasetForMILToolbox(D);
C = model;

% evaluation
a = genmil(a);
W = getdata(C);
[bag,baglab,bagid] = getbags(a);
n = size(bag,1);
SCB = zeros(n,1);
for i=1:n
    d = bag{i}*W.kernelmap;
    SCB(i,1) = max(+d(:,W.I),[],1)*W.w + W.w0;
end

PLB = double(SCB>0);



% s_out = sigm(PLB);

% w = prdataset([1-s_out s_out],baglab,'featlab',getlabels(C));
% w = setprior(w,getprior(a,0));
% w = setident(w,bagid,'milbag');





%% instance Classification


nb = length(D.B);
PL = D.Y*10;
SC = [];

for i=1:nb
    
    %% classify instance using the method in the paper
    % number of instance in the bag
    niib = sum(D.XtB==D.B(i));
    
    % find distance with every instances in the data set
    d = D.X(D.XtB==D.B(i),:)*W.kernelmap;
    % get the distance with only the selected data
    seld = +d(:,W.I);
    
    % find the 
    [maxProx, Us] = max(seld,[],1);
    
    
    % find the number of instance contributing to score for each
    % prototype (sometimes more than one instances can have equal distance)
    
    if niib > 1
        sortedDist = sort(seld,1,'descend');
        mk = zeros(length(W.I),1);
        for k = 1:size(sortedDist,2)
            tmp = 1;
            for kk = 2:size(sortedDist,1)
                if sortedDist(kk,k) == sortedDist(kk-1,k)
                    tmp = tmp+1;
                else
                    mk(k) = tmp;
                    break;
                end
            end
        end
    else
        mk = ones(length(W.I),1);
    end
    
    % compute score for each instance in the bag
    instScores = zeros(niib,1);
    
    % for each instance anf prototype combination
    for inst = 1:niib
        for proto = 1:length(W.I)
            % if the instance is the closest to the prototype
            if seld(inst,proto) == maxProx(proto) 
                instScores(inst) = instScores(inst) + ...
                    W.w(proto)*seld(inst,proto)/mk(proto);
            end
        end
    end
    
    % compute threshold for bag
    TH = -W.w0/length(unique(Us));
    IL =  instScores>TH;
    PL(D.XtB==D.B(i)) = IL;
    
    
    SC = [SC;instScores];
    
end


out.SCB = SCB;
out.PLB = PLB;
out.PL = PL;
out.SC = SC;




end

