function [pred] = trainAndTestMIL(D, DT, method, opt)
% INPUTS:
% method is a string containing the name of the MIL methods to use
% D is a MILdataset object containing the training set
% DT is a MILdataset object containing the test set
% OUTPUT:
% pred is an object containing:
%   The scores (SC) for intsance and bags (SCB)
%   The predicted labels (PL) for instances and bags (PLB)
%   The ground truth for instance (TL) and bags (TLB)

switch method
    
    case 'EMDD'
        model = EMDD(D,'train',[],opt);
        pred =  EMDD(DT,'test',model,opt);
        
    case 'MILES'
        model = myMILES(D,'train',[]);
        pred =  myMILES(DT,'test',model);
        
    case 'RSIS'
        model = RSIS(D,'train',[],opt);
        pred =  RSIS(DT,'test',model,opt);
        
    case {'kNN','SIkNN'}
        model = SIkNN(D,'train',[]);
        pred =  SIkNN(DT,'test',model);
        
    case 'MILBoost'
        model = myMILBoost(D,'train',[],opt);
        pred =  myMILBoost(DT,'test',model,opt);
        
    case {'CKNN','CkNN','C-KNN','C-kNN'}
        model = myCKNN(D,[],'train',[]);
        pred =  myCKNN(D,DT,'test',model);
        
    case 'CCE'
        model = CCE(D,'train',[]);
        pred =  CCE(DT,'test',model);
        
    case 'miSVM'
        model = miSVM(D,'train',[],opt);
        pred =  miSVM(DT,'test',model);
        
    case {'SISVM','SI-SVM'}
        model = SISVM(D,'train',[],opt);
        pred =  SISVM(DT,'test',model);
        
    case 'MI-SVM'
        model = MMISVM(D,'train',[],opt);
        pred =  MMISVM(DT,'test',model);
        
    case {'MI-kernel','NSK-SVM'}
        model = NSKSVM(D,'train',[]);
        pred =  NSKSVM(DT,'test',model);
        
    case 'EMD-kernel'
        model = EMDkernel(D,DT,'train',[],opt);
        pred =  EMDkernel(D,DT,'test',model,opt);
        
    case 'migraph'
        model = miGraph(D,'train',[]);
        pred =  miGraph(DT,'test',model);
        
    case 'BoW'
        model = BOW(D,[],'train',[],opt);
        pred =  BOW(D,DT,'test',model,[]);
        
    case {'SI-SVM-TH','SISVMTH'}
        model = SISVMTH(D,'train',[],opt);
        pred =  SISVMTH(DT,'test',model);
        
    case {'MInD'}
        model = MInD(D,DT,'train',[],opt);
        pred =  MInD(D,DT,'test',model,opt);
            
    otherwise
        error('Do not know this method')
end

end