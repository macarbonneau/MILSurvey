%LNSRCH  Simulate the routine "lnsrch" in [1], which takes,
%      xold   - The starting point of lnsrch
%      n      - Dimension of the instance
%      fold   - The value of function at xold
%      g      - The gradient of function at xold
%      p      - The direction for lnsrch
%      tolx   - Convergence tolerance on delta x
%      stpmax - stpmax is an input quantity that limits the length of the steps so that you do not try to valuate the function in regions where
%               it is undefined or subject to overflow.
%      and returns,
%      xnew   - The ending point of lnsrch
%      fnew   - The value of function at xnew
%      check  - Indicator which is 0 for normal exit, 1 when xnew is too close to xold
%
%    For more details, see [1]
%    [1] Press W H, Teukolsky S A, Vetterling W T, Flannery B P. Numerical Recipes in C: the art of scientific computing. Cambrige University Press,  
%        New York, 2nd Edition, 1992
    
function [xnew,fnew,check]=marc_lnsrch(xold,n,fold,g,p,tolx,stpmax,bags,baglabs)

%monitor = [];

    ALF=1e-4;
    check=0;
    sum=sqrt(p*p');
    if(sum>stpmax)
       p=p*(stpmax/sum);  %Scale if attempted step is too big
    end
    slope=g*p';
    if(slope>=0)
        %disp(strcat('Roundoff problem in lnsrch, slope(',num2str(slope),')is not negative:'));
        %dd_message(5,'Roundoff problem in lnsrch, slope(%f)is not negative:',slope);
%         disp('negative slope')
        xnew=xold;
        fnew=fold;
        check=1;
        return;
    end
    test=max(abs(p)./max(abs(xold),1));   %Compute minimum lamda
    if(test==0)
        xnew=xold;
        fnew=fold;
        check=1;
        return;
    end
    alamin=tolx/test;
    alam=1;   %Always try full Newton step first
    while(1)  %Start of iteration loop
        xnew=xold+alam*p;
       
        fnew=log_DD_here(xnew,bags,baglabs);
        
        %monitor = [monitor alam];
        if(alam<alamin)   %Convergence on delta x
            xnew=xold;
            check=1;
            %plot(monitor)
            %pause
            return;
        else
            if(fnew<=fold+ALF*alam*slope)  %Sufficient function decrease
                %plot(monitor)
                %pause
                return;
            else                           %Backtrack
                if (alam==1)
                    tmplam=-slope/(2*(fnew-fold-slope));   %First time
                else                                       %Subsequent backtracks
                    rhs1=fnew-fold-alam*slope;
                    rhs2=f2-fold-alam2*slope;
                    if(alam==alam2)
                        check=1;
                        xnew=xold;
                        fnew=fold;
                        return;
                    end
                    a=((rhs1/(alam*alam))-(rhs2/(alam2*alam2)))/(alam-alam2);
                    b=(((-alam2*rhs1)/(alam*alam))+((alam*rhs2)/(alam2*alam2)))/(alam-alam2);
                    if(a==0)
                        tmplam=-slope/(2*b);
                    else
                        disc=b*b-3*a*slope;
                        if(disc<0)
                            tmplam=0.5*alam;
                        else
                            if(b<=0)
                                tmplam=(-b+sqrt(disc))/(3*a);
                            else
                                tmplam=-slope/(b+sqrt(disc));
                            end
                        end
                    end
                    if(tmplam>0.5*alam)
                        tmplam=0.5*alam;
                    end
                end
                alam2=alam;
                f2=fnew;
                alam=max(tmplam,0.1*alam);
            end
        end
    end
    
                    
end

function [p] = log_DD_here(pars,bags,baglabs)

n = size(baglabs,1);
d2 = size(bags{1},2);
concept = pars(1:d2);
s = pars((d2+1):end);

prob = zeros(n,1);

for i=1:n
	[prob(i)] = bagprob_here(bags{i},baglabs(i),concept,s);
end
prob = max(prob,1e-12);
p = -sum(log(prob));

end
                    

%
function [bagp] = bagprob_here(bag,lab,concept,s)

m = size(bag,1);
dff = bag - repmat(concept,m,1);
dff2 = dff.^2;
s2 = s.^2;
% first the probability:
p = exp( -dff2*s2' );
p1minp = prod(1-p); %DXD: here at least 1 high detection will give 0

if lab>0
	% here is the OR function:
	bagp = 1-p1minp;
else
	bagp = p1minp;
end

end
                    
        
            
            
            
        
            
            
