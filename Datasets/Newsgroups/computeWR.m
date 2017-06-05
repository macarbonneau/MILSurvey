function [] = computeWR()

clear all
close all
clc

allDataset = dir('*.txt');

for d = 1:length(allDataset)
    
    disp(['TREATING: ' allDataset(d).name])
    
    fid = fopen(allDataset(d).name);
    D = getDataFromFile(fid);
    fclose(fid);
    
  WR(d) = getWR(D);
      
end

WR
mean(WR)

end


function [D] = getDataFromFile(fid)


D = MILdataset;

frmt = repmat('%f ',1,204);

tmp = textscan(fid,frmt,'CommentStyle','%','CollectOutput',true);
tmp = tmp{1};
D.X = tmp(:,5:204);
D.Y = tmp(:,2);
D.XtB = tmp(:,1);
D.YR = double(tmp(:,4)>0);
D.B = unique(D.XtB);

D.YB = D.B*0;

for b = 1:length(D.B)
    D.YB(b) = double(sum(D.YR(D.B(b)==D.XtB))>0);
end

end


function WR = getWR(D)


for b = 1:length(D.B)
   
    if D.YB(b) == 1
    idx = D.XtB==D.B(b);
    tmp = D.YR(idx);
    WR(b) = mean(tmp);
    end
end

WR = mean(WR)

end