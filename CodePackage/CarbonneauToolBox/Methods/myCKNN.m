function out =  myCKNN(D,DT,operation,model)

switch operation
    
    case 'train'
        
        [out.cit, out.ref] = validationCKNN(D);
        
    case 'test'
        
        [PB, NB] = convertDatasetForZhouToolbox(D);
        [~,~,TB] = convertDatasetForZhouToolbox(DT);
        
        % classify bags
        [out.PLB, out.SCB] = CKNNtest_Fast(PB,NB,TB,model.ref,model.cit,1);
        
        
        % classify instances 
        TB = convertDatasetForZhouToolbox_intances(DT);
        %[out.PL, out.SC] = CKNNtest(PB,NB,TB,model.ref,model.cit,1);
        [out.PL, out.SC] = CKNNtest_Fast(PB,NB,TB,model.ref,model.cit,1);
        
        % remove prediction from bags that were classified as negative
        [out.PL, out.SC] = adjustInstancePred(DT,out.PL,out.SC,out.PLB);
             
        % give true labels
        out.TL = DT.YR;
        out.TLB = DT.YB;     
end

end



function [refs, citers] = validationCKNN(D)

refs = [1 3 5 7 9 11 13 15];
citers = [1 3 5 7 9 11 13 15];

% compute distance matrix
DM = createDistanceMatrix(D.X);


nFolds = 5;
tic

% matrix for saving results
AUC = zeros(length(refs),length(citers),nFolds);

BagPerFoldList = divideBagsInFolds(nFolds,D);

for f = 1:nFolds
    
    statusBar(f,nFolds)
    D.X = 1:size(D.X,1);
    D.X = D.X';
    [TRD, TED] = getTrainingAndTestDatasets(f,nFolds,BagPerFoldList,D);
    
    
    [PB, NB] = convertDatasetForZhouToolbox_here(TRD);
    [~,~,testBags] = convertDatasetForZhouToolbox_here(TED);
    
    [PL, SC] = CKNN(PB,NB,testBags,refs,citers,1,DM);
    
    for r = 1:length(refs)
        for c = 1:length(citers)
            
            AUC(r,c,f) = getAUROC(SC(:,c,r), TED.YB, 0.1);
        end
    end
end

AUC = mean(AUC,3);

% get the best performing parameters
[~,ind] = max(AUC(:));
[c,g] = ind2sub(size(AUC),ind);

disp(['the best config for CKNN is : refs=' num2str(refs(c)) ' and citers=' num2str(citers(g))])


refs = refs(c);
citers = citers(g);
toc

end

function dist = hausdroffInstance(DM,iB1,iB2)

subDM = DM(iB1,iB2);
dist = min(subDM(:));
end

function [Pbags, Nbags, Abags] = convertDatasetForZhouToolbox_here(D)

Pbags = cell(1);
Nbags = cell(1);
Abags = cell(1);

% create positive bags
ind = D.YB == 1;
list = D.B(ind);
for i = 1:length(list)
    
    sel = D.XtB == list(i);
    Pbags{i} = D.X(sel);
end

% create negative bags
ind = D.YB == 0;
list = D.B(ind);
for i = 1:length(list)
    
    sel = D.XtB == list(i);
    Nbags{i} = D.X(sel);
end

list = D.B;
for i = 1:length(list)
    sel = D.XtB == list(i);
    Abags{i} = D.X(sel);
end


Pbags = Pbags';
Nbags = Nbags';

end

function [labels, scores] = CKNN(PBags,NBags,testBags,Refs,Citers,indicator,DM)
%  CKNN  Using the Citation KNN algorithm[1] to get the labels for bags in testBags, where minmum Hausdorff distance is used to measure the distances between bags
%     CKNN takes,
%        PBags     - an Mx1 cell array where the jth instance of ith positive bag is stored in PBags{i}(j,:)
%        NBags     - an Nx1 cell array where the jth instance of ith negative bag is stored in NBags{i}(j,:)
%        testBags  - a Kx1 cell array where the jth instance of ith test bag is stored in testBags{i}(j,:)
%         Refs     - the number of referecnes for each test bag
%        Citers    - the number of citers for each test bag
%        indicator - when the nearest neighbours have equal number of postive bags and negative bags, if indicator==1,then the output
%                    label is 1, otherwise 0, default=0
%
%     and returns,
%        labels    - a Kx1 vector which correspond to the output label of each bag in testBags
%
% For more details, please reference to bibliography [1]
% [1] J. Wang and J.-D. Zucker. Solving the multiple-instance problem: a lazy learning approach. In: Proceedings of the 17th
%     International Conference on Machine Learning, San Francisco, CA: Morgan Kaufmann, 1119-1125, 2000.


num_pbags=length(PBags);
num_nbags=length(NBags);
num_testbags=length(testBags);

if((Refs>num_pbags+num_nbags)|(Citers>num_pbags+num_nbags))
    error('too many Refs or Citers');
end

labels=zeros(num_testbags,length(Citers),length(Refs));
scores=zeros(num_testbags,length(Citers),length(Refs));


for i=1:num_testbags
    
    % disp(['CKNN TestBag ' num2str(i) ' of ' num2str(num_testbags)])
    
    num_bags=num_pbags+num_nbags+1;
    dist=-eye(num_bags);   %for every bag in testBAgs, initialize the distance matrix
    
    for j=1:num_bags
        if(j==1)
            for k=(j+1):(num_pbags+1)
                
                %                 dist(j,k) = myHausdorffDist_mex(testBags(i,:),PBags{k-1});
                dist(j,k) = hausdroffInstance(DM,testBags{i},PBags{k-1});
                dist(k,j)=dist(j,k);
                
            end
            for k=(num_pbags+2):num_bags
                
                %                 dist(j,k)= myHausdorffDist_mex(testBags(i,:),NBags{k-num_pbags-1});
                dist(j,k) = hausdroffInstance(DM,testBags{i},NBags{k-num_pbags-1});
                dist(k,j)=dist(j,k);
            end
        else
            if((j>=2)&(j<=num_pbags+1))
                for k=(j+1):(num_pbags+1)
                    %                     dist(j,k)=myHausdorffDist_mex(PBags{j-1},PBags{k-1});
                    dist(j,k) = hausdroffInstance(DM,PBags{j-1},PBags{k-1});
                    dist(k,j)=dist(j,k);
                end
                for k=(num_pbags+2):num_bags
                    %                     dist(j,k)=myHausdorffDist_mex(PBags{j-1},NBags{k-num_pbags-1});
                    dist(j,k) = hausdroffInstance(DM,PBags{j-1},NBags{k-num_pbags-1});
                    dist(k,j)=dist(j,k);
                end
            else
                for k=(j+1):num_bags
                    %                     dist(j,k)=myHausdorffDist_mex(NBags{j-num_pbags-1},NBags{k-num_pbags-1});
                    dist(j,k) = hausdroffInstance(DM,NBags{j-num_pbags-1},NBags{k-num_pbags-1});
                    dist(k,j)=dist(j,k);
                end
            end
        end
    end
    
    
    
    for ic = 1:length(Citers)
        for ir = 1:length(Refs)
            
            rp=0;   %get the references and citers of current test bag
            rn=0;
            cp=0;
            cn=0;
            
            [~,index]=sort(dist(1,:));
            for ref=1:Refs(ir)
                if(index(ref+1)<=num_pbags+1)
                    rp=rp+1;
                else
                    rn=rn+1;
                end
            end
            for pointer=2:(num_pbags+1)
                [~,index]=sort(dist(pointer,:));
                if(find(index==1)<=Citers(ic)+1)
                    cp=cp+1;
                end
            end
            
            
            for pointer=(num_pbags+2):num_bags
                [~,index]=sort(dist(pointer,:));
                if(find(index==1)<=Citers(ic)+1)
                    cn=cn+1;
                end
            end
            
            pos=rp+cp;
            neg=rn+cn;
            
            scores(i,ic,ir) = pos/(pos+neg);
            
            if(pos>neg)
                labels(i,ic,ir)=1;
            else
                if(pos==neg)
                    if(indicator==1)
                        labels(i,ic,ir)=1;
                    end
                end
            end
        end
    end
    
end

end

function [labels, scores] = CKNNtest(PBags,NBags,testBags,Refs,Citers,indicator)
%  CKNN  Using the Citation KNN algorithm[1] to get the labels for bags in testBags, where minmum Hausdorff distance is used to measure the distances between bags
%     CKNN takes,
%        PBags     - an Mx1 cell array where the jth instance of ith positive bag is stored in PBags{i}(j,:)
%        NBags     - an Nx1 cell array where the jth instance of ith negative bag is stored in NBags{i}(j,:)
%        testBags  - a Kx1 cell array where the jth instance of ith test bag is stored in testBags{i}(j,:)
%         Refs     - the number of referecnes for each test bag
%        Citers    - the number of citers for each test bag
%        indicator - when the nearest neighbours have equal number of postive bags and negative bags, if indicator==1,then the output
%                    label is 1, otherwise 0, default=0
%
%     and returns,
%        labels    - a Kx1 vector which correspond to the output label of each bag in testBags
%
% For more details, please reference to bibliography [1]
% [1] J. Wang and J.-D. Zucker. Solving the multiple-instance problem: a lazy learning approach. In: Proceedings of the 17th
%     International Conference on Machine Learning, San Francisco, CA: Morgan Kaufmann, 1119-1125, 2000.


num_pbags=length(PBags);
num_nbags=length(NBags);
num_testbags=length(testBags);

if((Refs>num_pbags+num_nbags)|(Citers>num_pbags+num_nbags))
    error('too many Refs or Citers');
end

labels=zeros(num_testbags,1);
scores=zeros(num_testbags,1);


for i=1:num_testbags
    
    % disp(['CKNN TestBag ' num2str(i) ' of ' num2str(num_testbags)])
    
    num_bags=num_pbags+num_nbags+1;
    dist=-eye(num_bags);   %for every bag in testBAgs, initialize the distance matrix
    
    for j=1:num_bags
        if(j==1)
            for k=(j+1):(num_pbags+1)
                
                dist(j,k) = myHausdorffDist_mex(testBags{i},PBags{k-1});
                %dist(j,k) = hausdroffInstance(DM,testBags{i},PBags{k-1});
                dist(k,j)=dist(j,k);
                
            end
            for k=(num_pbags+2):num_bags
                
                dist(j,k)= myHausdorffDist_mex(testBags{i},NBags{k-num_pbags-1});
                %dist(j,k) = hausdroffInstance(DM,testBags{i},NBags{k-num_pbags-1});
                dist(k,j)=dist(j,k);
            end
        else
            if((j>=2)&(j<=num_pbags+1))
                for k=(j+1):(num_pbags+1)
                    dist(j,k)=myHausdorffDist_mex(PBags{j-1},PBags{k-1});
                    %   dist(j,k) = hausdroffInstance(DM,PBags{j-1},PBags{k-1});
                    dist(k,j)=dist(j,k);
                end
                for k=(num_pbags+2):num_bags
                    dist(j,k)=myHausdorffDist_mex(PBags{j-1},NBags{k-num_pbags-1});
                    %dist(j,k) = hausdroffInstance(DM,PBags{j-1},NBags{k-num_pbags-1});
                    dist(k,j)=dist(j,k);
                end
            else
                for k=(j+1):num_bags
                    dist(j,k)=myHausdorffDist_mex(NBags{j-num_pbags-1},NBags{k-num_pbags-1});
                    %dist(j,k) = hausdroffInstance(DM,NBags{j-num_pbags-1},NBags{k-num_pbags-1});
                    dist(k,j)=dist(j,k);
                end
            end
        end
    end
    
    
    
    
    rp=0;   %get the references and citers of current test bag
    rn=0;
    cp=0;
    cn=0;
    
    [~,index]=sort(dist(1,:));
    for ref=1:Refs
        if(index(ref+1)<=num_pbags+1)
            rp=rp+1;
        else
            rn=rn+1;
        end
    end
    for pointer=2:(num_pbags+1)
        [~,index]=sort(dist(pointer,:));
        if(find(index==1)<=Citers+1)
            cp=cp+1;
        end
    end
    
    
    for pointer=(num_pbags+2):num_bags
        [~,index]=sort(dist(pointer,:));
        if(find(index==1)<=Citers+1)
            cn=cn+1;
        end
    end
    
    pos=rp+cp;
    neg=rn+cn;
    
    scores(i) = pos/(pos+neg);
    
    if(pos>neg)
        labels(i)=1;
    else
        if(pos==neg)
            if(indicator==1)
                labels(i)=1;
            end
        end
    end
    
end

end

function I = convertDatasetForZhouToolbox_intances(D)

I = cell(size(D.X,1),1);

for i = 1:size(D.X,1)
    I{i} = D.X(i,:);
end
% I = I';
end

function [labels, scores] = CKNNtest_Fast(PBags,NBags,testBags,nRefs,nCiters,indicator)
%  CKNN  Using the Citation KNN algorithm[1] to get the labels for bags in testBags, where minmum Hausdorff distance is used to measure the distances between bags
%     CKNN takes,
%        PBags     - an Mx1 cell array where the jth instance of ith positive bag is stored in PBags{i}(j,:)
%        NBags     - an Nx1 cell array where the jth instance of ith negative bag is stored in NBags{i}(j,:)
%        testBags  - a Kx1 cell array where the jth instance of ith test bag is stored in testBags{i}(j,:)
%         Refs     - the number of referecnes for each test bag
%        Citers    - the number of citers for each test bag
%        indicator - when the nearest neighbours have equal number of postive bags and negative bags, if indicator==1,then the output
%                    label is 1, otherwise 0, default=0
%
%     and returns,
%        labels    - a Kx1 vector which correspond to the output label of each bag in testBags
%
nPB=length(PBags);
nNB=length(NBags);
nTB=length(testBags);

idxP = 1:length(PBags);
idxN = (1:length(NBags))+nPB;
idxT = (1:length(testBags))+nPB+nNB;
idxTR = [idxP idxN];

DM = buildDistanceMatrixBetweenBags([PBags;NBags;testBags]);




if((nRefs>nPB+nNB)|(nCiters>nPB+nNB))
    error('too many Refs or Citers');
end

labels=zeros(nTB,1);
scores=zeros(nTB,1);

   
    
for i = 1:length(idxT)    

    rp=0;   %get the references and citers of current test bag
    rn=0;
    cp=0;
    cn=0;
    
    %%% find the training bags closest to the test bag (the references)
    
    tmpDist = DM(idxT(i),idxTR); % get correct entries in the distance matrix    
    [~,sortIdx]=sort(tmpDist);  % sort distance by closeness    
    closestBagIndex = idxTR(sortIdx); % arrange all bag indexes
    
    % compile votes for references
    for ref=1:nRefs
        
        if(closestBagIndex(ref)<=nPB)
            rp=rp+1;
        else
            rn=rn+1;
        end
    end
    
    %%% find the citers
    for cit = idxP
       tmpDist = DM(cit,[idxTR idxT(i)]);
       [~,sortIdx] = sort(tmpDist);
       sortIdx = sortIdx(1:nCiters);
  
       if any(sortIdx==nPB+nNB+1);
            cp=cp+1;
       end
    end
    
    for cit = idxN
       tmpDist = DM(cit,[idxTR idxT(i)]);
       [~,sortIdx] = sort(tmpDist);
       sortIdx = sortIdx(1:nCiters);
       
       if any(sortIdx==nPB+nNB+1);
            cn=cn+1;
       end
    end
    
    
    pos=rp+cp;
    neg=rn+cn;
 
    scores(i) = pos/(pos+neg);
    
    if(pos>neg)
        labels(i)=1;
    else
        if(pos==neg)
            if(indicator==1)
                labels(i)=1;
            end
        end
    end
    
end

end

function DM = buildDistanceMatrixBetweenBags(B)

DM = inf(length(B));

for i = 1:length(B)
    for j = i+1:length(B)
       DM(i,j) = myHausdorffDist_mex(B{i},B{j});
       DM(j,i) = DM(i,j);
    end
end

end

function [PL, SC] = adjustInstancePred(D,PL,SC,PLB)

for b = 1:length(D.B)
   
    if PLB(b) == 0; 
        idx = D.XtB == D.B(b);
        PL(idx) = 0;
        SC(idx) = 0;
    end
    
end

end