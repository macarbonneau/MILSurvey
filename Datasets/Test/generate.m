D = MILdataset;

for b = 1:20
    
    D.B = [D.B ;b];
    D.YB = [D.YB ;b>10];
    
    tmp = [rand(5,30) repmat(b>10,5,1)];
    D.X = [D.X; tmp];
    D.YR = [D.YR; repmat(b>10,5,1)];
    D.Y = D.YR;
    D.XtB = [D.XtB; repmat(b,5,1)];
end

DT = D;

save('testCV','D');
save('test','D','DT');