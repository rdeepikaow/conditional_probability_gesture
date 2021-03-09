function [cleaned_trace] = clean_trace(raw_trace)
% Code to clean the tracked peaks trace and output only the smoothed trace
% remove outliers
if (isempty(raw_trace))
    cleaned_trace = raw_trace;
    return;
end
nan_indices = isnan(raw_trace);
non_nan_indices = ~isnan(raw_trace);
raw_trace_without_nan = raw_trace;  raw_trace_without_nan(nan_indices) = [];
[~,outliers] = hampel(1:length(raw_trace_without_nan),raw_trace_without_nan,5);
raw_trace_without_nan(outliers) = nan;
raw_trace(non_nan_indices) = raw_trace_without_nan;

figure; plot(raw_trace); hold on;grid on;
%% Look for horizontal segments i.e., same peak location for several S locations
horizontal_th = 200;
datalength = length(raw_trace);
stop_index = datalength;
for i = 1:datalength
    data_segment = raw_trace(i:min(i+horizontal_th,datalength));
    data_segment(isnan(data_segment)) = [];
    if range(data_segment)<horizontal_th/5
        stop_index = i;
        break;
    end
end

cleaned_trace = raw_trace(1:min(stop_index+horizontal_th,datalength));

%% Discontinuity checking in x dimension
raw_trace = cleaned_trace;
raw_trace_nans = isnan(raw_trace);
raw_trace_nans_sum = movsum(raw_trace_nans,100);
discontinuities = find(raw_trace_nans_sum==100,1,'first');
if (~isempty(discontinuities))
    cleaned_trace = raw_trace(1:max(1,discontinuities-50));
end

%% Discontinuity checking in y dimension
raw_trace = cleaned_trace;
% moving window of 20
stop_checking_index = find(~isnan(cleaned_trace),1,'last');
start_checking_index = find(~isnan(cleaned_trace),1,'first');

stop_index = stop_checking_index;
for i = start_checking_index:stop_checking_index
    data_segment = raw_trace(i:min(stop_checking_index,i+40));
%     if isnan(max(diff(data_segment)))
%         stop_index = i;
%         break;
%     end
    data_segment(isnan(data_segment)) = [];
    [~,outliers] = hampel(1:length(data_segment),data_segment,2);
    data_segment(outliers) = [];
    data_segment = movmean(data_segment,3);
    if (max(abs(diff(data_segment)))>150)
        stop_index = i;
        break;
    end
end
cleaned_trace = raw_trace(1:min(stop_checking_index,stop_index+40));


figure; plot(cleaned_trace); hold on;grid on;
end


