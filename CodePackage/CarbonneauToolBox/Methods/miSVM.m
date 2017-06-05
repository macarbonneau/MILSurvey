function out =  miSVM(D,operation,model,opt)


switch operation
    case 'train'
                
        Y = double(D.Y);   
                
        for r = 1:opt.nIter
            
            Yold = Y;
            % train SVM
            [model] = trainSVM_CV(D.X,Y,opt);
            
            % get new labels
            
            % classify update labels using new model
            [Ytmp, ~, DV] = svmpredict(Y,D.X, model);
            [DV] = correctDV(DV,Ytmp);
            Y = double(logical(Y) & logical(Ytmp));
            
            % make sure at least one instance is positive in each bag
            Y = correctPosBags(D,Y,DV);
            
            if Y==Yold
                disp(['optmization finished after ' num2str(r) ' iterations'])
                break
            end
            
        end
        
        out = model;
        
    case 'test'
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % classify update labels model
        [out.PL, ~, out.SC] = svmpredict(D.YR,D.X, model);
        
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);
        
end

end


function Ynew = correctPosBags(D,Ynew,DV)

sb = D.YB == 1;
sb = D.B(sb);

for i = 1:length(sb)
    
    sel = D.XtB == sb(i);
    if sum(Ynew(sel)) == 0;
        
        [val] = max(DV(sel));
        isel = DV == val;
        Ynew(isel) = 1;
        
    end
    
end
end

function [DV] = correctDV(DV,PL)
sp = sum(DV(PL==1));
sn = sum(DV(PL~=1));

if sp < sn
    DV = -DV;
end

end