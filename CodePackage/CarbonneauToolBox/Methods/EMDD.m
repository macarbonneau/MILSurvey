function out =  EMDD(D,operation,model,opt)


switch operation
    case 'train'
  
        out = trainEMDD(D,opt);
                  
    case 'test'
        
        out = classifyEMDD(D,model);
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);
end

end


function [out] = trainEMDD(D,opt)


% INTPUT
%     D      MIL dataset
%     NP     number of prototype used for selection

% PARAMETERS
%     FRAC   Fraction/number of instances taken into account in
%            evaluation (frac = 1)
%     K      Number of objects (or fraction of objects when K<1) used as
%            random initialisation (default = 10)
%     EPOCHS Number of runs in the maxDD (default = [4 4])
%     TOL    Tolerances (default = [1e-5 1e-5 1e-7 1e-7])
%
%
% Use the Expectation Maximization version of the Maximum Diverse
% Density. It is an iterative EM algorithm, requiring a sensible
% initialisation. By giving ALF or K, you can specify how many times you
% would like to run the algorithm. From the K tries the best one (on the
% training set) is returned.
%
% SEE ALSO
% maxDD_mil


%% initialize with parameters
x = convertDatasetForMILToolbox(D);
%frac = 1;
% number of prototype
NP = opt.NP;
tol = opt.tol;
epochs = opt.epochs;


%% Extract the bags, find the bags
[bags,baglab,~,~] = getbags(x);

nrbags = length(bags);
bagI = ispositive(baglab);

dim = size(x,2);
epochs = epochs*dim;

%% define how many (and which) points are used for initialization
startpoint = bags(find(bagI));
startpoint = cell2mat(startpoint);
I = randperm(size(startpoint,1));
if (NP<1) % fraction
    k = max(round(NP*length(I)),1);
else
    k = NP;
end

if k>size(startpoint,1)
    warning('mil:emdd_mil',...
        'Asking for too many starting points, just use all data');
    k = size(startpoint,1);
else
    startpoint = startpoint(I(1:k),:);
end


%%
%NOTE: magic number here: normalize data to unit variance before!
scales = repmat(0.1,k,dim);
pointlogp = inf(k,1);


%% optimization k times:
parfor i=1:k
    bestinst = cell(nrbags,1);
    logp1 = log_DD_here([startpoint(i,:),scales(i,:)],bags,bagI);
    
    
    disp(['PROTOTYPE INIT ' num2str(i) '/' num2str(k) ' ->'])
    disp('==========|')
    tic
    % do a few runs to optimize the concept and scales in an EM
    % fashion:

    for r=1:10    
        % find the best fitting instance per bag
        for j=1:nrbags
            dff = bags{j}-repmat(startpoint(i,:),size(bags{j},1),1);
            dff = (dff.^2)*(scales(i,:)').^2;
            [mn,J] = min(dff);
            bestinst{j} = bags{j}(J,:);
        end
        % run the maxDD on only the best instances
        maxConcept = maxdd_here(startpoint(i,:),scales(i,:),...
            bestinst,bagI,epochs,tol);
        startpoint(i,:) = maxConcept{1,1}(1:dim);
        scales(i,:) = maxConcept{1,1}(dim+1:end);
        % do we improve?
        logp0 = logp1;
        logp1 = log_DD_here(maxConcept{1,1},bags,bagI);
        % disp(['avant ' num2str(logp0)])
        disp(['logP ' num2str(logp1)])
        if abs(exp(-logp1)-exp(-logp0))<0.01*exp(-logp0)
            break;
        end
        fprintf('*')
    end
    toc
    
    %dd_message(5,'. \n');
    pointlogp(i) = logp1;
end

%% find best DD from all initialisation
% now we did it k times, what is the best one?
[mn,J] = min(pointlogp);
out.concept = [startpoint(J,:), scales(J,:)];

end



function out = classifyEMDD(D,model)

%% get probabilities for all instances with best point

PI = zeros(size(D.X,1),1); % probabiblity for all instances
for i = 1:size(D.X,1)
    PI(i) = InstanceDD(model.concept,D.X(i,:));
end

out.SC = PI;
out.PL = double(PI>0.5);



end


function [prob] = InstanceDD(prototype,IUT)

d2 = size(IUT,2);
location = prototype(1:d2);
scale = prototype((d2+1):end);

dff = IUT - location;
dff2 = dff.^2;
s2 = scale.^2;

% the probability:
prob = exp( -dff2*s2' );
prob = max(prob,1e-15);

end


function [p,der] = log_DD_here(pars,bags,baglabs)

dim = size(pars,2);
n = size(baglabs,1);
d2 = size(bags{1},2);
concept = pars(1:d2);
s = pars((d2+1):end);

prob = zeros(n,1);
der = zeros(n,dim);
for i=1:n
    [prob(i), der(i,:)] = bagprob(bags{i},baglabs(i),concept,s);
end
prob = max(prob,1e-12);
der = -(1./prob)'*der;
p = -sum(log(prob));

return
end

%MAXDD The optimization of Diverse Density
%
%  [MAXcONCEPT,CONCEPTS] = MAXDD(SPOINTS,SCALES,BAGS,BAGI,EPOCHS,TOL)
%
% The core optimization function of maxDD_mil. See maxDD_mil.

function [maxConcept,concepts] = maxdd_here(spoints,scales,bags,bagI,epochs,tol)

% initialize some parameters and storage
num_start_points = size(spoints,1);
dim = size(spoints,2);
concepts = cell(num_start_points,2);
maxConcept{1} = [zeros(1,dim) ones(1,dim)];
maxConcept{2} = 0;

% monitor1 = [];
% monitor2 = [];
% monitor4 = [];

% make several runs, starting with another startingpoint spoint.
for i=1:num_start_points
    if num_start_points>1,
        dd_message(6,'%d/%d ',i,num_start_points);
    end
    xold = [spoints(i,:),scales];
    % compute the data likelihood and its derivative
    [fold g] = log_DD_here(xold,bags,bagI);
    p = -g;
    sumx = xold*xold';
    stpmax = 100*max(sqrt(sumx),2*dim); %upper bound on step size
    % now do an iterative line-search to find the global minimum
    crotte = zeros(epochs(1),1);
    crotte2 = zeros(epochs(1),1);
    for iter = 1:epochs(1)
        
        [xnew,fnew,check] = marc_lnsrch(xold,dim,fold,g,p,tol(3),stpmax,bags,bagI);
        % check if the step in space is still large enough:
        xi = xnew-xold;
        tst = max(abs(xi)./max(abs(xnew),1));
        
%         monitor1 = [monitor1 tst];
        if tst<tol(1)
            break
        end
        % check if the likelihood is changing sufficiently
        [dummy,g] = log_DD(xnew,bags,bagI);
        den = max(fnew,1);
        tst = max(abs(g).*max(abs(xnew),1))/den;
        
%         monitor2 = [monitor2 tst];
        if tst<tol(2)
            break;
        end
        % OK, store for the next step
        p = -g;
        xold = xnew;
        fold = fnew;
        sumx = xold*xold';
        stpmax = 100*max(sqrt(sumx),2*dim);
    end

    [xnew,fret,iterations(i,2)] = dfpmin_here(xnew,dim,tol(3),tol(4),epochs(2),bags,bagI);
    concepts{i,1} = xnew;
    concepts{i,2} = exp(-fret);
    
    if concepts{i,2}>maxConcept{2}
        maxConcept{1} = concepts{i,1};
        maxConcept{2} = concepts{i,2};
    end
end

%             subplot(2,2,2)
%             plot(monitor2)
%             %ylim([0,0.001])
%             
%             subplot(2,2,1)
%             plot(monitor1)
%             %ylim([0,0.001])
%             pause

end

%MIL_DFPMIN  Simulate the routine "dfpmin" in [1], which takes,
%     xold  - The starting point of dfpmin
%     n     - Dimension of the instance
%     tolx  - Convergence tolerance on delta x
%     gtol  - Convergence tolerance on gradient
%     itmax - Maximum allowed number of iterations
%     and returns,
%     xnew  - The ending point of dfpmin
%     fret  - The value of function at xnew
%     iter  - Number of iterations that were performed
%
%    For more details, see [1]
%    [1] Press W H, Teukolsky S A, Vetterling W T, Flannery B P. Numerical Recipes in C: the art of scientific computing. Cambrige University Press,
%        New York, 2nd Edition, 1992

function [xnew,fret,iter]=dfpmin_here(xold,n,tolx,gtol,itmax,bags,baglabs)

xnew=xold;
%fp=neg_log_DD(xold(1:n),xold((n+1):2*n));   %Caculate staring function value
%g=D_neg_log_DD(xold(1:n),xold((n+1):2*n));  %Caculate initial gradient
[fp,g] = log_DD(xold,bags,baglabs);
hessin=eye(2*n);   %Initialize the inverse Hessian to the unit matrix
xi=-g;
sum=xold*xold';
stpmax=100*max(sqrt(sum),2*n);

% crotte = zeros(itmax,1);
% crotte2 = zeros(itmax,1);
for its=1:itmax
    iter=its;
    [pnew,fret,check]=marc_lnsrch(xnew,n,fp,g,xi,tolx,stpmax,bags,baglabs);
    fp=fret;
    xi=pnew-xnew;   %Update the line direction
    xnew=pnew;      %Update the current pint
    test=max(abs(xi)./max(abs(xnew),1));         %Test for convergence on delta x
   

    if(test<tolx)
        break;
    end

       
    dg=g;    %Save the old gradient
    %g=D_neg_log_DD(xnew(1:n),xnew((n+1):2*n));  %Get the new gradient
    [dummy,g] = log_DD_here(xnew,bags,baglabs);
    den=max(fret,1);  %Test for convergence on zero gradient
    test=max(abs(g).*max(abs(xnew),1))/den;
    if(test<gtol)
        break;
    end

    dg=g-dg;   %Compute difference of gradients
    hdg=hessin*dg';  %Compute difference times current matrix
    fac=dg*xi';       %Calculate dot products for the denominators
    fae=dg*hdg;
    sumdg=dg*dg';
    sumxi=xi*xi';
    if(fac>sqrt(3e-8*sumdg*sumxi))   %Skip update if fac not sufficiently positive
        fac=1/fac;
        fad=1/fae;
        dg=fac*xi-fad*hdg';   %The vector that makes BFGS different from DFP
        hessin=hessin+fac*(xi'*xi)-fad*(hdg*hdg')+fae*(g'*g);
    end
    xi=(-hessin*g')';
    
end



end
