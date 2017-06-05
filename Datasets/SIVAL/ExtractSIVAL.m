function [] = ExtractSIVAL()

clear all
close all
clc

allDataset = dir('*.data');

for d = 1:length(allDataset)
    
    disp(['TREATING: ' allDataset(d).name])
    
    fid = fopen(allDataset(d).name);
    D = getDataFromFile(fid);
    fclose(fid);
    
    fn = allDataset(d).name;
    fn = strrep(fn,'.data','');
    fn = ['SIVAL-subsampled- ' fn];
    save(fn,'D');

    
end

end


function [D] = getDataFromFile(fid)

D = MILdataset;

%% recupertate data from file
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

end


