function [] = statusBar(i,n)
% this function diplays a status bar of 20 steps based on i/n


if n >= 20
    
    if i == 1
        disp('====================|')
    end
    
    if mod(i,floor(n/20)) == 0
        fprintf('*')
    end
    
    if i == n
        fprintf('\n')
    end
    
else
    if i == 1
        tmp = repmat('=',1,n);
        disp([tmp '|'])
    end
    
    fprintf('*')
 
    if i == n
        fprintf('\n')
    end
    
end


end