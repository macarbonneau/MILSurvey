function [] = StatsOnBirds()

clear all
close all
clc


for i = 1:13
    disp('====================================================')
 fn = ['Datasets/OneShot/Birds-V' num2str(i)]; 
 load(fn)
 disp(fn)
 
 tmp = sum(D.YB)/length(D.YB)*100;
 disp(['Percentage of positive bags: ' num2str(tmp)])
 
 tmp = sum(D.YR);
 disp(['positive instances: ' num2str(tmp)])
 
 WR = getBagWR(D);
 disp(['the WR is between:' num2str(WR(1)) ' and '  num2str(WR(2))])
 disp(['mean: ' num2str(WR(3)) ' median: ' num2str(WR(4))])
 
 
end
end


function out = getBagWR(D)

low = 100;
high = 0;
avr = 0;
ctr = 0;
WRV = [];

for i = 1:length(D.YB)
   if D.YB(i) == 1
      tmp = D.YR(D.XtB==D.B(i));
       WR = sum(tmp)/length(tmp)*100;
       if WR > high
           high = WR;
       elseif WR < low;
       low = WR;
       end
       
       avr = avr+WR;
       ctr = ctr+1;
       WRV = [WRV WR];    
   end
end

out = [low high avr/ctr median(WRV)];

end