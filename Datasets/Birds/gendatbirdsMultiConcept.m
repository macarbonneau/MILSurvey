%GENDATBIRDS Bird song MIL problem
%
%     A = GENDATBIRDS(NR)
%
% INPUT
%   NR     Target class (default = 1)
%
% OUTPUT
%   A      MIL dataset
%
% DESCRIPTION
% Define the MIL problem of identifying bird songs. This is a
% multi-class problem. Each bag is a recording of one or more birds. The
% bag inherits all the labels of the birds present in the recording.
% This can be converted to a two-class problem by choosing a bird target
% class. The bird classes are:
%
%  1	BRCR - Brown Creeper
%  2	WIWR - Winter Wren
%  3	PSFL - Pacific-slope Flycatcher
%  4	RBNU - Red-breasted Nuthatch
%  5	DEJU - Dark-eyed Junco
%  6	OSFL - Olive-sided Flycatcher
%  7	HETH - Hermit Thrush
%  8	CBCH - Chestnut-backed Chickadee
%  9	VATH - Varied Thrush
%  10	HEWA - Hermit Warbler
%  11	SWTH - Swainson's Thrush
%  12	HAFL - Hammond's Flycatcher
%  13	WETA - Western Tanager    
%
% Each bag (10 second recording) is converted to a spectogram, and a
% segmentation procedure is applied. An instance is represented by a
% segment of a spectogram and is described by 38 features (shape of the
% segment, its time and frequency profile statistics, histogram of
% gradients).  An instance is labelled either with one of the 13
% classes, or with -1 to indicate that no label is given. 
%
% REFERENCE
% @inproceedings{briggs2012rank,
%  title={Rank-loss support instance machines for MIML instance annotation},
%  author={Briggs, F. and Fern, X.Z. and Raich, R.},
%  booktitle={Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining},
%  pages={534--542},
%  year={2012},
%  organization={ACM}
% }
%
% Here two versions of the datasets are used: filtered (only labeled
% instances) and unfiltered (all instances)
%
% SEE ALSO
% mildatapath

 
function D = gendatbirdsMultiConcept(targetcl,removeNoise)


D = MILdataset;



mildata = [];
millab = [];
milbagid = [];


%% Get Features and segments labels

%Get features and instance labels
featid = fopen('hja_birdsong_features.txt');
featline = fgetl(featid); %Line with description
featline = fgetl(featid); %Instance 1

instlabid = fopen('/hja_birdsong_instance_labels.txt');
instlabline = fgetl(instlabid); 
instlabline = fgetl(instlabid); 

i=1;  
while ischar(featline) 
    
        %Features of segment
        feats = str2num(featline);
        
        %Bag ID and instance label
        instlabdata = str2num(instlabline); 
        
        bagid = instlabdata(1);
        instlab = instlabdata(2);
       if ~(removeNoise &&  instlab==-1)
        %Add to dataset / labels
        mildata = [mildata; feats(:,2:end)];
        millab = [millab; instlab];   %This is the actual instance label, as a class number. -1 means no label
        milbagid = [milbagid; bagid]; %Record from which bag the instance is
       end
    featline = fgetl(featid);
    instlabline = fgetl(instlabid);
   
    i=i+1;
end


fclose(featid);
fclose(instlabid);

%% save into the MIL data set object
D.X = mildata;

idx = millab == targetcl(1);
for i = 2:length(targetcl)
tmp = millab == targetcl(i);
idx = idx|tmp;
end
    
%% Get info based on the bags
D.YR = double(idx);
D.XtB = milbagid;

D.B = unique(D.XtB);
D.YB = D.B*inf;
D.Y = D.YR*inf;
for i = 1:length(D.B)
   
    idx = D.XtB == D.B(i);
    D.YB(i) = double(sum(D.YR(idx))>0);
    D.Y(idx) = D.YB(i);    
end

end

