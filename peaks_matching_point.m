function potential_peaks = peaks_matching_point(signal)
%% Function detects local peaks and returns the peak information for the direction function
N = length(signal);    % length of the signal
L = 50;
peaks = [];
for index = 1 : N
    % peaks
    if signal(index) > max(signal([max(index-L, 1) : index-1, index+1 : min(index+L, N)]))+eps
        peaks = [peaks, [index; signal(index)]];
    end
    % valleys (with a negative sign)
    if signal(index)+eps < min(signal([max(index-L, 1) : index-1, index+1 : min(index+L, N)]))
        peaks = [peaks, [-index; signal(index)]];
    end
end

%% Filter the significant peaks
potential_peaks = [];
if peaks(1,1)>0 && peaks(1,2)<0 && peaks(2,1)-peaks(2,2)>0.001
    potential_peaks = [potential_peaks , peaks(:,1)];
end
for p = 2:length(peaks)-1
    if peaks(1,p)>0 && peaks(1,p-1)<0 && peaks(1,p+1)<0
        left_significance = peaks(2,p)-peaks(2,p-1);
        right_significance  = peaks(2,p)-peaks(2,p+1);
        if max(left_significance,right_significance)>0.001
            potential_peaks = [potential_peaks, peaks(:,p)];
        end
    end
end

if peaks(1,end)>0 && peaks(1,end-1)<0 && peaks(2,end)-peaks(2,end-1)>0.001
    potential_peaks = [potential_peaks , peaks(:,end)];
end

