function [SC] = SIkNNClassify(X,Y,XT,k)

SC = zeros(size(XT,1),k);


for i = 1:size(XT,1)
   
    kdist = inf(k,1);
    kLab = inf(k,1);
    
    
    for j = 1:size(X,1)
        
        tmp = sum((X(j,:)-XT(i,:)).^2);
        
        if tmp < kdist(k)
           
            kdist(k) = tmp;
            kLab(k) = Y(j);
            [kdist, idx] = sort(kdist);
            kLab = kLab(idx);

        end        
    end
    
    for j = 1:k
       SC(i,j) = sum(kLab(1:j))/j; 
    end
    
end




end




function d = distH(x1,x2)

    d = sum((x1-x2).^2)

end

