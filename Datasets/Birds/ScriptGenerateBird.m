clc
clear all
close all

minWR = inf(1,13);
maxWR = zeros(1,13);
meanWR = zeros(1,13);

minIPB = inf(1,13);
maxIPB = zeros(1,13);
meanIPB = zeros(1,13);


for c = 1:13
    tic
   D = gendatbirds(c,0);
   D = normalizeUnitVarianceMIL(D, D);
   fn = ['Birds-C' num2str(c)];
   save(fn,'D');
   toc
   
   for i = 1:length(D.B)
      WR = D.YR(D.B(i) == D.XtB);
      IPB = length(WR);
      WR = sum(WR)/length(WR);
      
      if D.YB(i) == 1
         if WR == 0
            error('oups') 
         end
      end
      
      if WR == 1
         disp(['WR1 Bag ' num2str(i)]) 
      end
      
      
      if D.YB(i) == 1
      if WR < minWR(c)
          minWR(c) = WR;
      end
      if WR > maxWR(c)
          maxWR(c) = WR;
      end
      meanWR(c) = meanWR(c)+WR;
      end
      
      if IPB < minIPB(c)
          minIPB(c) = IPB;
      end
      if IPB > maxIPB(c)
          maxIPB(c) = IPB;
      end
      meanIPB(c) = meanIPB(c)+IPB;
      
      
       
   end
   meanWR(c) = meanWR(c)/sum(D.YB);
   meanIPB(c) = meanIPB(c)/length(D.B);
   
   
   
   disp(['For Class ' num2str(c) '-----------------'])
   disp(['min: ' num2str(minWR(c)) '   max: ' num2str(maxWR(c)) ...
       '   mean ' num2str(meanWR(c))])
   
   
   disp(['min: ' num2str(minIPB(c)) '   max: ' num2str(maxIPB(c)) ...
       '   mean ' num2str(meanIPB(c))])
   
   
end
