function distance = myHausdorffDist(Bag1,Bag2)
% minHausdorff  Compute the minimum Hausdorff distance between two bags Bag1 and Bag2
% minHausdorff takes,
%   Bag1 - one bag of instances
%   Bag2 - the other bag of instanes
%   and returns,
%   distance - the minimum Hausdorff distance between Bag1 and Bag2

 
    line_num1=size(Bag1,1);
    line_num2=size(Bag2,1);
    
    
    dist=zeros(line_num1,line_num2);
    for i=1:line_num1
        for j=1:line_num2
            dist(i,j)=sqrt(sum((Bag1(i,:)-Bag2(j,:)).^2));
        end
    end
    
    tmp = zeros(size(dist,1),1);
    for i = 1:size(dist,1)
        tmp(i) = min(dist(i,:)); 
    end
    
    distance=min(tmp);

        
end