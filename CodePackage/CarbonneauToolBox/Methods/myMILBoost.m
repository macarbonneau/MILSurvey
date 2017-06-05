function out =  myMILBoost2(D,operation,model,opt)


switch operation
    case 'train'
        
        out = trainMILBoost(D,opt);
        
    case 'test'
        
        out = MyMILBoostClassify(D,model);
        
end

end

function model = trainMILBoost(D,opt)

opts = optimset('fminunc');
opts = optimset(opts,'Display','off','LargeScale','off');

errtol = 1e-15;

nB = length(D.B);
T = opt.boostingRounds;

a = zeros(T,1);
h = zeros(T,3);
w = zeros(size(D.X,1),1);
prev_out = ones(size(D.X,1),1);


plotV = [];
plotV2 = [];

sig = @(x) 1./(1+exp(-x));

for t=1:T
    
    %% compute weights for each bag
    pij = sig(-prev_out);
    pi = zeros(nB,1);
    for i=1:nB
        idx = D.XtB==D.B(i);
        %pi(i) = 1 - prod(1-pij(idx));
        pi(i) = max(pij(idx));
        if D.YB(i)==1
            w(idx) = (1-pi(i))*pij(idx)/pi(i);
        else
            w(idx) = -pij(idx);
        end
    end
    
    % I run into problems when the weights are virtually zero, so avoid that
    % it really becomes too small:
    tol = 1e-6;
    I = find(abs(w)<tol);
    if ~isempty(I)
        sgn = sign(w(I));
        sgn = (w(I)>=0)*2-1;
        w(I) = sgn.*tol;
    end
    
    %% train weak classifier
    [h(t,:),besterr] = traindecstump(D.X,w);
    
    %% this hyposthesis gives
    this_out = h(t,3)*sign(D.X(:,h(t,1))-h(t,2));
    
    %% get best alpha 
    a(t) = fminunc(@(a) lossFunction(a,prev_out,this_out,D),0,opts);
    
    %% update classifier response
    prev_out = prev_out + a(t)*this_out;
    
    %% check if we should continue
    if (besterr<=errtol), break; end
end
%% Re-adjust stuff if we stopped early
if (t<T)
    T=t;
    h=h(1:T,:);
    a=a(1:T,:);
end

%% save model
model.a = a;
model.h = h;

end



function logL = lossFunction(alpha,prev_out,this_out,D)


nB = length(D.B);

pij = 1./(1+exp(-prev_out-alpha*this_out));
pi = zeros(nB,1);
logL = 0;


% run over the bags
for i=1:nB
    idx = D.XtB==D.B(i);
    
    pi(i) = 1 - prod(1-pij(idx));
    
    if D.YB(i)==1
        logL = logL - log(pi(i)+eps);
    else
        logL = logL - log(1-pi(i)+eps);
    end
end


% for i=1:nB
%     idx = D.XtB==D.B(i);
%     
%     pi(i) = max(pij(idx));
%     
%     if D.YB(i)==1
%         logL = logL - log(pi(i)+eps);
%     else
%         logL = logL - log(1-pi(i)+eps);
%     end
% end

end


function [h,besterr] = traindecstump(x,w)

[n,dim] = size(x);

sumneg = (w<=0)'*w;
sumpos = (w>0)'*w;
besterr = inf;
bestfeat = 1;

bestthr = 0;
bestsgn = 0;

for i=1:dim
    % find the best threshold for feature i
    % assume that the positive class is on the right of the decision
    % threshold:
    [sx,J] = sort(x(:,i));
    z = cumsum(w(J));
    
    err1 = -sumneg + z;
    [minerr,I] = min(err1);
    if (minerr<besterr)
        besterr = minerr;
        bestfeat = i;
        if (I==n)
            bestthr = sx(I)+10*eps;
        else
            bestthr = (sx(I)+sx(I+1))/2;
        end
        bestsgn = +1;
    end
    
    % Now assume that the positive class is on the left of the decision
    % threshold:
    err2 =  sumpos - z;
    [minerr,I] = min(err2);
    if (minerr<besterr)
        besterr = minerr;
        bestfeat = i;
        if (I==n)
            bestthr = sx(I)+10*eps;
        else
            bestthr = (sx(I)+sx(I+1))/2 + eps;
        end
        bestsgn = -1;
    end
    
end

h = [bestfeat bestthr bestsgn];

end









function out = MyMILBoostClassify(D,model)

% give true labels
out.TL = D.YR;
out.TLB = D.YB;

SC = zeros(size(D.X,1),1);

for i = 1:length(model.a)
    SC = SC + model.a(i)*model.h(i,3)*sign(D.X(:,model.h(i,1))-model.h(i,2));
end

pij = 1./(1+exp(-SC));

SCB = zeros(length(D.B),1);
for i=1:length(D.B)
    idx = D.XtB==D.B(i);
    SCB(i) = max(pij(idx));
end


out.SC = SC;
out.SCB = SCB;

out.PLB = out.SCB >= 0.5;
out.PL = out.SC >=0;

end

