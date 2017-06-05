function [Pbags, Nbags, Abags, AbagsY] = convertDatasetForZhouToolbox(D)

Pbags = cell(1);
Nbags = cell(1);
Abags = cell(1);
AbagsY = zeros(length(D.B),1);

% create positive bags
ind = D.YB == 1; 
list = D.B(ind);
for i = 1:length(list)
   
    sel = D.XtB == list(i);
    Pbags{i} = D.X(sel,:);  
end

% create negative bags
ind = D.YB == 0; 
list = D.B(ind);
for i = 1:length(list)
    
    sel = D.XtB == list(i);
    Nbags{i} = D.X(sel,:);  
end


% create cell for all bags and the corresonding label vector
for i = 1:length(D.B)
    
    sel = D.XtB == D.B(i);
    Abags{i} = D.X(sel,:);
    AbagsY(i) = D.YB(i);

end

Pbags = Pbags';
Nbags = Nbags';
Abags = Abags';


end