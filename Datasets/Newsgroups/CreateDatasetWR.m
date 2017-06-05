function [] = CreateDatasetWR()

close all
clc

% Number of bags
NB = 100;
% Instances per bags
IPB = 100;
% Probability of postive instances in positive bags
% (there is always at least one)
WR = [1 5 10 50 75 100];

%% Data Set creation

for rep = 1:10
    
    %% Instance pool creation
    disp(['Version ' num2str(rep)])
    [POS,NEG] = getAllInstances();
    
    % generate training set only negative
    [D, DT] = sampleNegDataSets(NB,IPB,NEG);
    
    for w = 1:length(WR)
        
        [D,POS] = adjustWitnessRate(WR(w),POS,D);
        [DT,POS] = adjustWitnessRate(WR(w),POS,DT);
        
        % Save
        fileName = ['Particles-WR' num2str(WR(w)) '-V' num2str(rep)];
        disp(fileName)
        save(fileName,'D','DT');
       
    end
    
end

for i = 1:size(D.X,2)-1

    %subplot(2,5,i)
    figure
    scatter(D.X(logical(D.Y),i),D.X(logical(D.Y),i+1),5,'r');
    hold on
    scatter(D.X(~logical(D.Y),i),D.X(~logical(D.Y),i+1),5,'+b');
    hold on
    scatter(DT.X(~logical(DT.Y),i),DT.X(~logical(DT.Y),i+1),5,'+g');
    hold on
    scatter(DT.X(logical(DT.Y),i),DT.X(logical(DT.Y),i+1),5,'c');
end

end

function [POS,NEG] = getAllInstances()

%% load data set
Xp = dlmread('PositiveClass1000');
Xn = dlmread('NegativeClass');

X = [Xp;Xn];
Y = [ones(size(Xp,1),1); zeros(size(Xn,1),1)];

X = normalizeUnitVariance(X, []);

% create instance inventory
POS = X(Y==1,:);
NEG = X(Y~=1,:);

clear X
clear Xp
clear Xn

end


function [D,DT] = sampleNegDataSets(NB,IPB,NEG)

D = MILdataset;%

D.B = 1:NB;
D.B = D.B';
DT = D;

%% create bags Data sets containing only negative instances
D.X = NEG(1:NB*IPB,:);
NEG(1:NB*IPB,:) = [];
DT.X = NEG(1:NB*IPB,:);
NEG(1:NB*IPB,:) = [];


for b = 1:NB
    
    %%%%%%%%%%%%%%%%%% if a positive bag
    if mod(b,2) == 0
        
        D.YB = [D.YB;1];
        DT.YB = [DT.YB;1];
        
        D.Y = [D.Y;ones(IPB,1)];
        D.YR = [D.YR;zeros(IPB,1)];
        D.XtB = [D.XtB;ones(IPB,1)*b];
        
        DT.Y = [DT.Y;ones(IPB,1)];
        DT.YR = [DT.YR;zeros(IPB,1)];
        DT.XtB = [DT.XtB;ones(IPB,1)*b];
        
    else
        %%%%%%%%%%%%%%%%%%% is a negative bag %%%%%%%%%%%%%%%
        D.YB = [D.YB;0];
        DT.YB = [DT.YB;0];
        
        D.Y = [D.Y;zeros(IPB,1)];
        D.YR = [D.YR;zeros(IPB,1)];
        D.XtB = [D.XtB;ones(IPB,1)*b];
        
        DT.Y = [DT.Y;zeros(IPB,1)];
        DT.YR = [DT.YR;zeros(IPB,1)];
        DT.XtB = [DT.XtB;ones(IPB,1)*b];
        
    end
    
end

end




function [D,POS] = adjustWitnessRate(WR,POS,D)


%% create bags Data sets complete seen or unseen
for b = 1:length(D.B)
    
    % if a positive bag
    if D.YB(b) == 1
             
        idx = D.XtB==D.B(b);
        
        Xtmp = D.X(idx,:);
        YRtmp = D.YR(idx,:);
        
        % compute target positive
        TPI = max(1,round(length(YRtmp)*WR/100));
        NPI = 0;
        for i = 1:length(YRtmp)
            
            if NPI < TPI
                
                if YRtmp(i) ~= 1
                    YRtmp(i) = 1;
                    Xtmp(i,:) = POS(1,:);
                    POS(1,:) = [];
                end              
                NPI = NPI+1;
            else
                break;
            end
        end
        
        D.X(idx,:) = Xtmp;
        D.YR(idx,:) = YRtmp;
        
    end
end

disp(['WR: ' num2str(sum(D.YR)/length(D.YR)*100) ' % '])


end



function [mu, sig, enD] = createConcept(NC, D, IDP, covEn)

if covEn % if covariance is enabled
    
    mu = rand([NC,D])*6-3;
    sig = getCovMatrix(D,NC);
    
    % dimension enabled in the dataset
    nEnD = ceil((1-IDP)*D);
    enD = false(NC,D);
    for i = 1:NC
        idx = randperm(D);
        idx = idx(1:nEnD);
        enD(i,idx) = true;
    end
    
else % if covariance is not enabled
    
    mu = rand([NC,D])*6-3;
    sig = rand([NC,D])/10;
    
    % dimension enabled in the dataset
    nEnD = ceil((1-IDP)*D);
    enD = false(NC,D);
    for i = 1:NC
        idx = randperm(D);
        idx = idx(1:nEnD);
        enD(i,idx) = true;
    end
    
end
end

function inst = sampleInstanceFromConcept(mu, sig, enD, nD, NC, NDS, covEn)

inst = rand([1,nD])*NDS-NDS/2;

if covEn % if covariance is enabled
    
    c = randi(NC);
    tmp = mvnrnd(squeeze(mu(c,:)),squeeze(sig(c,:,:)));
    inst(enD(c,:)) = tmp(enD(c,:));
    
else % if covariance is not enabled
    
    tmp = randn([1,nD]);
    c = randi(NC);
    tmp = tmp.*sig(c,:)+mu(c,:);
    inst(enD(c,:)) = tmp(enD(c,:));
    
end
end

function sig = getCovMatrix(D,NC)

sig = zeros(NC,D,D);

for i = 1:NC
    
    Q = randn(D,D);
    
    eigen_mean = 0; % can be made anything, even zero % used to shift the mode of the distribution
    
    tmp = (Q' * diag(abs(eigen_mean+rand(D,1)*.100)) * Q);
    
    d = diag(tmp);
    amp = max(d);
    tmp = tmp/amp*0.1;
    sig(i,:,:) = tmp;
    
end

end

