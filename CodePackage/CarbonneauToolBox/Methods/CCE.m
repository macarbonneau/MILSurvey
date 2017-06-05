function out =  CCE(D,operation,model)


switch operation
    case 'train'
        
        opt.NC = 11; % number of classifiers
        opt.ranNumClusters = false;
        opt.C = [1 10 100 1000 10000]; % for SVM
        opt.gamma = [0.01 0.1 1 10 100 1000];
        opt.kernel = [0 2];
        opt.metric = 'AUC';
        
        out = trainCCE(D, opt);
%         [out.cit, out.ref] = validationCCE(D,opt);
        
        
    case 'test'
        
        out = classifyCCE(D,model);
        
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % bogus labels for instance 
        out.PL = D.YR;
        out.SC = D.YR;
        
end

end


function model = trainCCE(D, opt)

[~, ~, TrB,TrY] = convertDatasetForZhouToolbox(D);

[num_train,~]=size(TrB);
en_size = opt.NC;
% [~,en_size]=size(num);

model = cell(opt.NC,2);

if opt.ranNumClusters
   num = randi(36,[en_size 1])+4; 
else
    num = 3:2:en_size*2+3;
end

instances=[];
for i=1:num_train
    instances=[instances;TrB{i,1}];
end

for iter=1:en_size
    disp(strcat('Building the ',num2str(iter),'-th individual classifier'));
    
    [centers,~] = vl_kmeans(instances',num(iter),'NumRepetitions',10);
    centers = centers';
    
    model{iter,1} = centers;
    
    trainset=zeros(num(iter),num_train);
    
    for i=1:num_train
        tempbag=TrB{i,1};
        [tempsize,~]=size(tempbag);
        for j=1:tempsize
            tempdist=zeros(1,num(iter));
            for k=1:num(iter)
                tempdist(1,k)=(tempbag(j,:)-centers(k,:))*(tempbag(j,:)-centers(k,:))';
            end
            [~,index]=min(tempdist);
            trainset(index,i)=trainset(index,i)+1;
        end
    end
    trainset=(trainset>=1);
    
    SVM = trainSVM_CV(double(trainset'),double(TrY),opt);
    model{iter,2} = SVM;
    
    
end

end

function [out]=classifyCCE(D,model)

[~, ~, TeB,TeY] = convertDatasetForZhouToolbox(D);
[num_test,~]=size(TeB);

Outputs=zeros(num_test,size(model,1));


for iter=1:size(model,1)

    nK = size(model{iter,1},1);
    testset=zeros(nK,num_test);

    
    for i=1:num_test
        tempbag=TeB{i,1};
        tempsize = size(tempbag,1);
        
        
        centers = model{iter,1};
        for j=1:tempsize
            tempdist=zeros(1,nK);
            for k=1:nK
                tempdist(1,k)=(tempbag(j,:)-centers(k,:))*(tempbag(j,:)-centers(k,:))';
            end
            [~,index]=min(tempdist);
            testset(index,i)=testset(index,i)+1;
        end
    end
    testset=(testset>=1);
    
    
    PY = svmpredict(TeY,double(testset'), model{iter,2});
    
    %     [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(trainset,traintarget,gamma,cost);
    %     Labels=ones(1,num_test);
    %     [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(testset, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
    Outputs(:,iter)=PY;
end


votes = sum(Outputs,2);
out.PLB = double(votes>size(model,1)/2);
out.SCB = votes;

end






