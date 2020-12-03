function [peaks_tracked_final] = DP_peak_tracking_forward(matrix,peaks,motion_stats,turn_index,peak_significance,debug)
params =[];
params.peak_displacement_th = 25;

discard_indices = find(peaks<5);
peaks_bak = peaks;
peaks(discard_indices) = [];
motion_stats_bak = motion_stats;
motion_stats(discard_indices) = 0;

%% Obtain initial point
modified_peaks = peaks-turn_index;
start_index = find(peaks-turn_index~=0,1,'last');

%% Function implements dynamic programming to track the peaks
% Costs:(a) Smoother curve incurs lower cost.
% (b) Deviation from the peak_locs will incur more cost.

% matrix = matrix(:,start_index:end);
% peaks = (1:length(peaks))-peaks+start_index-1;
if (debug)
    figure; imagesc(matrix);colorbar;hold on;grid on;
    plot(peaks_bak,1:length(peaks_bak),'r*'); hold on;grid on;
end
matrix_bak = matrix;
matrix(:,discard_indices) = [];
[L,N] = size(matrix);
total_cost = cell(1,N);
total_index = cell(1,N);
total_cost{start_index} = zeros(1,N);
total_index{start_index}=ones(1,N).*peaks(start_index);
for i = start_index-1:-1:1
    total_cost{i} = zeros(1,N);
    total_index{i} = zeros(1,N);
    current_cost = total_cost{i};
    current_index = total_index{i};
    previous_cost = total_cost{i+1};
    for j = 1:N
        min_cost = 1e20;min_index = 1;
        for k = 1:N
            if (modified_peaks(i)~=0 && peaks(i)~=0)
                %                 test_cost = 5*abs(k-j)./(motion(i)+eps) + abs(j-peaks(i)) + 20*abs(matrix(j,i)-matrix(peaks(i),i))  + previous_cost(k);
                test_cost = 20*abs(k-j) + abs(j-peaks(i)) + 20*abs(matrix(i,j)-matrix(i,peaks(i)))  + previous_cost(k);
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
for i = start_index:-1:1
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
peaks_tracked_flipped = fliplr(peaks_tracked);
difference_peak_trace = diff(peaks_tracked_flipped);
discontinuity_indices = find(abs(difference_peak_trace)>50,1);

%% Also check if the significance of the peaks is higher after this
peak_significance = peak_significance(turn_index:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% add if needed
if (~isempty(discontinuity_indices))
    peaks_tracked_continuous = nan(size(peaks_tracked_flipped));
    peaks_tracked_continuous(1:discontinuity_indices) = peaks_tracked_flipped(1:discontinuity_indices);
else
    peaks_tracked_continuous = peaks_tracked_flipped;
end
peaks_tracked_continuous = fliplr(peaks_tracked_continuous);
if (debug)
    figure; plot(peaks_tracked); hold on;grid on; plot(peaks_tracked_continuous,'b*'); hold on;grid on;
end
%% Remove detected peaks which are very far apart from the predicted peak trace
predicted_actual_difference = abs(peaks_tracked(end-length(peaks_tracked_continuous)+1:end)-peaks_tracked_continuous);
remove_indices = predicted_actual_difference>100;
peaks_tracked_continuous(remove_indices) = nan;

if (debug)
    figure; plot(peaks_tracked); hold on;grid on; plot(peaks_tracked_continuous,'b*'); hold on;grid on;
    figure;
    imagesc(matrix_bak); colorbar; hold on;grid on; plot(peaks_tracked_continuous,1:length(peaks_tracked_continuous),'b*'); hold on;grid on;
end

peaks_tracked_final = nan(size(peaks_tracked_continuous));
peaks_tracked_final(peaks_tracked_continuous>=5) = peaks_bak(peaks_tracked_continuous>=5);

% In peak estimation, the maximum location of the peak is taken as twice
% the distance between sample and turn location. Remove peaks which lie on
% the maximum value.
x = 1:length(peaks_tracked_final);
y = 2*turn_index-x+1;
distance_from_end = abs(peaks_tracked_final-y);
peaks_tracked_final(distance_from_end==0) = nan;
