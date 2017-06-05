function DM = createDistanceMatrix(X)

sx = size(X,1);
DM = zeros(sx);

for i = 1:sx
    
    for j = i+1:sx
        
        DM(i,j) = sum((X(i,:)-X(j,:)).^2);
        DM(j,i) = DM(i,j);
    end
end
end
