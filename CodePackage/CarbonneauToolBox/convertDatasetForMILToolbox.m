function OUT = convertDatasetForMILToolbox(D)

multiClass = false;
if length(unique(D.YB)) > 2
    multiClass = true;
end

%% two class

if ~multiClass
% convert labels
L = [];
for i = 1:length(D.Y)
    if D.Y(i) == 1
        L = [L; 'positive'];
    else
        L = [L; 'negative'];
    end
    
end

OUT = genmil(D.X,L,D.XtB,'presence');
OUT = setprior(OUT,[0.5 0.5]);

if ~ismilset(OUT,1) 
 disp('INCORRECT DATA')
end

else
   error('MULTI CLASS DATA SET SUPPLIED')
end

end


