function [peaks_tracked_final,fraction_of_segment] = DP_peak_tracking(matrix,peaks,motion_stats,turn_index,peak_significance,debug)
params =[];
params.peak_displacement_th = 25;

discard_indices = find(peaks<5);
peaks_bak = peaks;
peaks(discard_indices) = [];
motion_stats_bak = motion_stats;
motion_stats(discard_indices) = 0;

%% Obtain initial point
modified_peaks = (1:length(peaks))-peaks+turn_index-1;
start_index = find(modified_peaks~=0,1);

%% Function implements dynamic programming to track the peaks
% Costs:(a) Smoother curve incurs lower cost.
% (b) Deviation from the peak_locs will incur more cost.

% matrix = matrix(:,start_index:end);
% peaks = (1:length(peaks))-peaks+start_index-1;
if (debug)
    figure; imagesc(matrix);colorbar;hold on;grid on;
    plot(1:length(peaks_bak),peaks_bak,'r*'); hold on;grid on;
end
matrix_bak = matrix;
matrix(:,discard_indices) = [];
[L,N] = size(matrix);
total_cost = cell(1,N);
total_index = cell(1,N);
total_cost{start_index} = zeros(1,L);
total_index{start_index}=ones(1,L).*peaks(start_index);
for i = start_index+1:N
    total_cost{i} = zeros(1,L);
    total_index{i} = zeros(1,L);
    current_cost = total_cost{i};
    current_index = total_index{i};
    previous_cost = total_cost{i-1};
    for j = 1:L
        min_cost = 1e10;min_index = 1;
        for k = 1:L
            if (modified_peaks(i)~=0 && peaks(i)~=0)
                %                 test_cost = 5*abs(k-j)./(motion(i)+eps) + abs(j-peaks(i)) + 20*abs(matrix(j,i)-matrix(peaks(i),i))  + previous_cost(k);
                test_cost = 15*abs(k-j) + abs(j-peaks(i)) + 20*abs(matrix(j,i)-matrix(peaks(i),i))  + previous_cost(k);
            else
                %                 test_cost = 5*abs(k-j)./(motion(i)+eps) + previous_cost(k);
                test_cost = 15*abs(k-j)+ previous_cost(k);
            end
            if (test_cost<min_cost)
                min_cost = test_cost;
                min_index = k;
            end
        end
        current_cost(j) = min_cost;
        current_index(j) = min_index;
    end
    total_cost{i} = current_cost;
    total_index{i} = current_index;
end

%% Backtracking
peaks_tracked = nan(size(peaks));
final_min_cost = nan(size(peaks));
for i = 1:N
    if (~isempty(total_cost{i}))
        [min_cost,min_ind] = min(total_cost{i});
        final_min_cost(i) = min_cost;
        peaks_tracked(i) = total_index{i}(min_ind);
    end
end

correct_indices = setdiff(1:length(peaks_bak),discard_indices);
peaks_tracked_final = nan(1,length(peaks_bak));
peaks_tracked_final(correct_indices) = peaks_tracked;
peaks_tracked = peaks_tracked_final;


if (debug)
    plot(1:length(peaks_tracked),peaks_tracked,'b*'); hold on;grid on;
    figure; plot(peaks_bak); hold on;grid on; plot(peaks_tracked,'b*'); hold on;grid on;
end

%% Discontinue the peak trace at which there is a drastic jump
difference_peak_trace = diff(peaks_tracked);
discontinuity_indices = find(abs(difference_peak_trace)>100,1);

%% Also check if the significance of the peaks is higher after this
peak_significance = peak_significance(turn_index:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% add if needed
if (~isempty(discontinuity_indices))
    peaks_tracked_continuous = peaks_tracked(1:discontinuity_indices);
else
    peaks_tracked_continuous = peaks_tracked;
end
if (debug)
    figure; plot(peaks_tracked); hold on;grid on; plot(peaks_tracked_continuous,'b*'); hold on;grid on;
end
%% Remove detected peaks which are very far apart from the predicted peak trace
predicted_actual_difference = abs(peaks_tracked(1:length(peaks_tracked_continuous))-peaks_tracked_continuous);
remove_indices = predicted_actual_difference>100;
peaks_tracked_continuous(remove_indices) = nan;

if (debug)
    figure; plot(peaks_tracked); hold on;grid on; plot(peaks_tracked_continuous,'b*'); hold on;grid on;
    figure;
    imagesc(matrix_bak); colorbar; hold on;grid on; plot(peaks_tracked_continuous,'b*'); hold on;grid on;
end
%% Calculate the fraction of the segment with the peaks
peak_trace_end_point = find(~isnan(peaks_tracked_continuous),1,'last');
motion_stats = motion_stats_bak(turn_index:end);
motion_stats(motion_stats<0) = 0;
fraction_of_segment2 = sum(motion_stats(1:peak_trace_end_point))/sum(motion_stats);

%% First segment fraction
peak_trace_end_point = peaks_tracked_continuous(peak_trace_end_point);
motion_stats = motion_stats_bak(1:turn_index);
motion_stats(motion_stats<0) = 0;
fraction_of_segment1 = 1- sum(motion_stats(1:peak_trace_end_point))/sum(motion_stats);

fraction_of_segment = max(fraction_of_segment1,fraction_of_segment2);

peaks_tracked_final = nan(size(peaks_tracked_continuous));
peaks_tracked_final(peaks_tracked_continuous>=5) = peaks_bak(peaks_tracked_continuous>=5);

%% Remove the maximum peaks sample locations
x = 1:length(peaks_tracked_final);
y = turn_index - 2*x+2;
difference_from_end = abs(y-peaks_tracked_final);
peaks_tracked_final(difference_from_end==0) = nan;