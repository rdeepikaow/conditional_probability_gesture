function [strrs] = smoothTRRS(trrs,backward_flag)
% Function to smooth the TRRS matrix using "smooth" function
% backward_flag indicates whether to smooth backwards or forward.
% 1=> trrs(i,i:-1:1) ; 0=> trrs(i,i:end);
N = size(trrs,1);
strrs = eye(N);
if (backward_flag)
    for i = 1:N
        vector = trrs(i:-1:1,i);
        hvector = hampel(1:length(vector),vector,50);
        svector = smooth(hvector,0.3,'loess');
        strrs(i:-1:1,i) = svector;
    end
else
    for i = 1:N
        vector = trrs(i,i:end);
        hvector = hampel(1:length(vector),vector,50);
        svector = smooth(hvector,0.1,'loess');
        strrs(i,i,i:end) = svector;
    end
end
        

end

