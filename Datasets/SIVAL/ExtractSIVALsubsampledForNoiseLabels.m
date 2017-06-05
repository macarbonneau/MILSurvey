function [] = ExtractSIVALsubsampledForNoiseLabels()

clear all
close all
clc

allDataset = dir('*.data');

noizeLevel = 0:10:100;

for d = 1:length(allDataset)
    
    disp(['TREATING: ' allDataset(d).name])
    
    fid = fopen(allDataset(d).name);
    Dtot = getDataFromFile(fid);
    fclose(fid);
    
    % normalize data set
    Dtot = normalizeUnitVarianceMIL(Dtot);
    
    % split test
    BagPerFoldList = divideBagsInFolds(3,Dtot);
    [Di, DT] = getTrainingAndTestDatasets(1,3,BagPerFoldList,Dtot);
    
    fn = allDataset(d).name;
    fn = strrep(fn,'.data','');
    
    for i = 1:length(noizeLevel)
        
        D = putNoisyLabels(Di,noizeLevel(i));
        fn = ['SIVAL-Noise' num2str(noizeLevel(i)) '-V' num2str(d)];
        
        save(fn,'D','DT');
    end
    
end

end

function D = putNoisyLabels(D,nl)
rng(1)
idx = randperm(length(D.B));
NN = round(length(D.B)*nl/100);

for i = 1:NN
    b = D.B(idx(i));
    D.YB(idx(i)) = ~D.YB(idx(i));
    if D.YB(idx(i)) == 1
        D.Y(D.XtB==b) = 1;
        D.YR(D.XtB==b) = 1;
    else
        D.Y(D.XtB==b) = 0;
        D.YR(D.XtB==b) = 0;
    end
end

end


function [D] = getDataFromFile(fid)

D = MILdataset;

%% recuperate data from file
frmt = ['%s ' repmat('%f ',1,32)];
tmp = textscan(fid,frmt,'CommentStyle','%','CollectOutput',true,'delimiter',',');
XtBText = tmp{1};
BagNames = unique(XtBText);
tmp = tmp{2};
D.X = tmp(:,2:31);
D.YR = tmp(:,32);
D.XtB = ones(size(D.X,1),1);
D.B = 1:length(BagNames);
D.B = D.B';
IN = tmp(:,1);
b = 1;
D.Y = D.YR*0;

%% List all bags and establish correspondances

for i = 2:size(D.X,1)
    if IN(i-1) > IN(i) % new bag detected
        b = b+1;
    end
    D.XtB(i) = b;
end

%% Compute WR and create instance label based on bags
maxWr = 0;
minWr = 100;
for b = 1:length(D.B)
    D.YB(b) = double(sum(D.YR(D.B(b)==D.XtB))>0);
    wr = sum(D.YR(D.B(b)==D.XtB))/length(D.YR(D.B(b)==D.XtB))*100;
    
    D.Y(D.XtB==D.B(b)) = D.YB(b);
    
    if wr ~= 0
        if wr > maxWr
            maxWr = wr;
        elseif wr < minWr
            minWr = wr;
        end
    end
end
D.YB = D.YB';
disp(['Minimum witness rate is ' num2str(minWr)])
disp(['Maximum witness rate is ' num2str(maxWr)])

%% remove negative bags

% list of negative bags
NB = D.B(D.YB ~= 1);

% list of class
for i = 1:length(XtBText)
    str = XtBText{i};
    tmp = textscan(str,'%s %s %s','CollectOutput',false,'delimiter','_');
    XtBText(i) = tmp{2};
end
classList = unique(XtBText);

YBstr = cell(size(D.B));
for i = 1:length(D.B)
    tmp =  XtBText(D.B(i)==D.XtB);
    tmp = unique(tmp);
    YBstr{i} = tmp{1};
end


% identify postive class
PosC = XtBText(D.Y==1);
PosC = unique(PosC);
if length(PosC)>1
    error('problem!!!')
end
PosC = PosC{1};

selI = false(size(D.X,1),1);
selB = false(size(D.B,1),1);

for c = 1:length(classList)
    
    if ~strcmp(classList{c},PosC)
        
        tmp = strcmp(classList{c},XtBText);
        
        % list of bags from this class
        tmpList = unique(D.XtB(tmp));
        % sample some bags
        keptBags = randsample(tmpList,5);
        
        selItmp = D.XtB==keptBags(1);
        selBtmp = D.B==keptBags(1);
        for b = 2:length(keptBags)
            selItmp = selItmp | D.XtB==keptBags(b);
            selBtmp = selBtmp | D.B==keptBags(b);
        end
        
    else
        selItmp = D.Y==1;
        selBtmp = D.YB==1;
        
    end
    
    selI = selI | selItmp;
    selB = selB | selBtmp;
    
end

D.X = D.X(selI,:);
D.Y = D.Y(selI);
D.YR = D.YR(selI);
D.XtB = D.XtB(selI);
D.B = D.B(selB);
D.YB = D.YB(selB);

end


