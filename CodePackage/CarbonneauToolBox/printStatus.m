function printStatus(p,rep,fold,npop,t,iK,iRS,iDSS,vfold)

%% print status
disp('==================================================')
disp(['Repetition: ' num2str(rep) ' / ' num2str(p.nRep)])
disp(['Test Fold: ' num2str(fold) ' / ' num2str(p.nFolds)])
disp(['Pop: ' num2str(npop) ' / ' num2str(length(p.nPop))])
disp(['Temperature: ' num2str(t) ' / ' num2str(length(p.T))])
disp(['K-means: ' num2str(iK) ' / ' num2str(length(p.nK))])
disp(['Random Subspaces: ' num2str(iRS) ' / ' num2str(length(p.nRS))])
disp(['Dim per SS: ' num2str(iDSS) ' / ' num2str(length(p.nDSS))])
disp(['Validation Folds: ' num2str(vfold) ' / ' num2str(p.nVfolds)])
disp('---------------------------------------------------')


den = p.nRep*p.nFolds;
num = (rep-1)*p.nFolds + (fold-1);


disp(['COMPLETED TOTAL: ' num2str(num/den*100) ' %'])
disp('==================================================')


end