function [smatrix] = smoothSimilarity(matrix,backward_flag)
% Function to smooth the TRRS matrix using "smooth" function
% backward_flag indicates whether to smooth backwards or forward.
% 1=> trrs(i,i:-1:1) ; 0=> trrs(i,i:end);
N = size(matrix,1);
smatrix = eye(N);
if (backward_flag)
    for i = 1:N
        vector = matrix(i:-1:1,i);
        hvector = hampel(1:length(vector),vector,50);
        svector = smooth(hvector,0.3,'loess');
        smatrix(i:-1:1,i) = svector;
    end
else
    for i = 1:N
        vector = matrix(i,i:end);
        hvector = hampel(1:length(vector),vector,50);
        svector = smooth(hvector,0.1,'loess');
        smatrix(i,i:end) = svector;
    end
end
        

end

