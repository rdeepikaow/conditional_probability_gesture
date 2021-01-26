classdef TurnAngleEstimation
    % Class contains functions required for estimating the turn
    % angle in a gesture for 5 GHz.
    
    properties
        potential_peaks_;
        peak_val_;
        peak_loc_;
        peak_valb_;
        peak_locb_;
        peak_detection_window_;
        turn_index_;
        peak_tracking_max_queue_;
        peak_tracking_max_jump_;
        similarity_matrix_;
        similarity_matrix_smoothed_;
        debug_;
        valley_val_;
        valley_loc_;
        valley_valb_;
        valley_locb_;
        peak_valley_val_;
        matfilename_;
        time_diff_;
        ref_time_diff_;
        start_index_;
        stop_index_;
        motion_index_;
        sounding_rate_set_;
        sf_;
        previous_segment_end_point_;
        center_regions_;
        fraction_of_segment_one_;
        fraction_of_segment_two_;
        peak_peak_loc_;
        peak_significance_;
        tracked_peak_trace_;
        potential_peaks_forward_ ;
        peak_val_forward_;
        peak_loc_forward_ ;
        valley_val_forward_ ;
        valley_loc_forward_ ;
        peak_valley_val_forward_;
        similarity_matrix_forward_;
        peak_significance_forward_;
        tracked_peak_trace_forward_;
        tracked_peak_trace_backward_;
    end
    
    methods
        function obj = TurnAngleEstimation(params)
            obj.peak_detection_window_ = params.peak_detection_window;
            obj.peak_tracking_max_queue_ = params.peak_tracking_max_queue;
            obj.peak_tracking_max_jump_ = params.peak_tracking_max_jump;
            obj.debug_ = params.debug;
            obj.matfilename_ = params.matfilename;
            obj.similarity_matrix_ = params.similarity_matrix;
            obj.similarity_matrix_smoothed_ = params.similarity_matrix_smoothed;
            obj.sounding_rate_set_ = params.sounding_rate_set;
            obj.sf_ = params.sf;
        end
        
        function obj = quick_turn_angle_estimation(obj,start_index,stop_index, turn_index, cummulative_ms)
            obj.motion_index_ = cummulative_ms;
            obj.start_index_ = start_index;
            obj.stop_index_ = stop_index;
            obj.turn_index_ = turn_index;
            %% Find center point of second segment
            [pks,locs] = findpeaks(cummulative_ms);
            second_segment_center_start = find(cummulative_ms(turn_index:end)>0.5*pks(2),1);
            second_segment_center_end = find(cummulative_ms(turn_index:end)>0.5*pks(2),1,'last');
            second_segment_center_start = second_segment_center_start + turn_index-1;
            second_segment_center_end = second_segment_center_end + turn_index -1 ;
            
            similarity_matrix = obj.similarity_matrix_smoothed_(start_index:stop_index,start_index:stop_index);
            N = size(similarity_matrix,1);
            peak_to_valley = zeros(1,N);
            sample_to_peak = zeros(1,N);
            sample_to_valley = zeros(1,N);
            for i = second_segment_center_start:second_segment_center_end
                sample_to_valley(i) = similarity_matrix(turn_index,i);
                backward_vector = similarity_matrix(1:turn_index,i);
                [max_val,max_loc] = max(backward_vector);
                sample_to_peak(i) = max_val;
                peak_to_valley(i) = similarity_matrix(max_loc,turn_index);
            end
            sample_to_valley = hampel(1:length(sample_to_valley),sample_to_valley,10);
            peak_to_valley = hampel(1:length(peak_to_valley),peak_to_valley,10);
            sample_to_peak = hampel(1:length(sample_to_peak),sample_to_peak,10);
            
            figure; 
            subplot(3,1,1);
            plot(sample_to_valley); hold on;grid on; ylim([0.98 1]);
            title('sample to valley');
            subplot(3,1,2);
            plot(peak_to_valley); hold on;grid on; ylim([0.98 1]);
            title('peak to valley');
            subplot(3,1,3);
            plot(sample_to_peak); hold on;grid on; ylim([0.98 1]);
            title('sample to peak');
            
            %% Calculate the similarity triangle area
            lower_bound = min(sample_to_valley(sample_to_valley>0.9));
            figure; 
%             subplot(2,1,1);
            plot(sample_to_valley); hold on;grid on;
            plot(sample_to_peak); hold on;grid on;
            ylim([lower_bound 1]);
%             subplot(2,1,2);
            plot(peak_to_valley); hold on;grid on;
%             plot(sample_to_peak); hold on;grid on;
%             ylim([lower_bound 1]);
            upper_bound = (sample_to_peak(find(sample_to_peak>0.9,1,'first')));
            difference_SV = upper_bound-sample_to_valley(sample_to_valley~=0);
            difference_SV(difference_SV<0) = 0;
            difference_SP = sum(upper_bound-sample_to_peak(sample_to_peak~=0));
            difference_SP(difference_SP<0) = 0;            
            area_under_SV = sum(difference_SV);
            area_under_SP = sum(difference_SP);
%             area_under_PV = sum(peak_to_valley(peak_to_valley~=0)-lower_bound);
%             fraction_area_SV = area_under_SV/area_under_SP;
            fraction_area = area_under_SP/area_under_SV;
            disp(['Fraction SV: ', num2str(fraction_area)]);
%             disp(['Fraction PV: ', num2str(fraction_area_PV)]);            
        end
        
        
        function obj = peak_valley_similarity_backward(obj,start_index,stop_index,turn_index,cummulative_ms)
            obj.motion_index_ = cummulative_ms;
            obj.start_index_ = start_index;
            obj.stop_index_ = stop_index;
            obj.turn_index_ = turn_index;
            similarity_matrix = obj.similarity_matrix_smoothed_(start_index:stop_index,start_index:stop_index);
            N = size(similarity_matrix,1);
            potential_peaks = cell(1,N-turn_index+1);
            potential_peaks_significances = cell(1,N-turn_index+1);
            potential_valleys = cell(1,N-turn_index+1);
            peak_val = zeros(1,N);
            peak_significance = zeros(1,N);
            peak_loc = zeros(1,N);
            valley_val = zeros(1,N);
            valley_loc = zeros(1,N);
            peak_valley_val = zeros(1,N);
            
            for i = turn_index:N
                if (i==1310)
                    temp = 1;
                end
%                 peak_loc_max = min(max(ceil(2*(i-turn_index)),1),turn_index-1);
                vector = similarity_matrix(turn_index:-1:1,i);
%                 vector = smooth(vector,0.3,'rloess');
                [potential_peaks{i},potential_peaks_significances{i},potential_valleys{i}] = obj.peakAnalysis(vector);
                array_peaks = potential_peaks{i};
                if (~isempty(array_peaks))
                    [peak_val(i),max_loc] = max(array_peaks(:,2));
                    peak_significance(i) = potential_peaks_significances{i}(max_loc);
                    peak_loc(i) = turn_index-array_peaks(max_loc,1)+1;%-turn_index);
                    %%%%%% If peak is not detected near the turn,ignore%%%
%                     if 2*abs(i-turn_index)<(turn_index-peak_loc(i))
%                         peak_loc(i) = 0;
%                         peak_significance(i)= 0 ;
%                         continue;
%                     end
                    %% If peak not detected at the end of segment,ignore%%
%                     if 0.5*abs(i-turn_index)>abs(peak_loc(i)-turn_index)
%                         peak_loc(i) = 0;
%                         peak_significance(i) = 0;
%                         continue;
%                     end
%                     
                    
                    
                    [valley_val(i),valley_loc(i)] = min(similarity_matrix(i:-1:peak_loc(i),i));
                    %% New addition:
                    valley_loc(i) = obj.turn_index_;
                    [peak_val(i),peak_loc(i)] = max(vector);
                    peak_loc(i) = turn_index-peak_loc(i)+1;
                    valley_val(i) = similarity_matrix(obj.turn_index_,i);
                    %valley_loc(i) = i-valley_loc(i)+1;%-turn_index);
                    peak_valley_val(i) = similarity_matrix(peak_loc(i),valley_loc(i));
%                     if (turn_index-peak_loc(i)<i-turn_index+50)
%                         peak_loc(i) = 0; peak_val(i) = 0;
%                         peak_significance(i) = 0;peak_valley_val(i)=0;
%                         valley_loc(i) = 0 ; valley_val(i) = 0;
%                         continue;
%                     end
                    
                end
            end
            
            obj.potential_peaks_ = potential_peaks;
            obj.peak_val_ = peak_val;
            obj.peak_loc_ = peak_loc;
            obj.valley_val_ = valley_val;
            obj.valley_loc_ = valley_loc;
            obj.peak_valley_val_ = peak_valley_val;
            obj.similarity_matrix_ = similarity_matrix;
            obj.peak_significance_ = peak_significance;
            
            if(obj.debug_)
                figure;
                subplot(2,1,1);
                plot((peak_val)); hold on;grid on;
                plot((peak_valley_val));hold on;grid on;
                legend('peak val','valley val');
                
                subplot(2,1,2);
                plot(peak_loc); hold on;grid on;
                plot((valley_loc)); hold on;grid on;
                legend('peak loc','valley loc');
            end
            
            
            obj = obj.peakTracking();
        end
        
        
        function obj = peakTracking(obj)
            %% Costs for dynamic programming
            % Deviation(in location) from the peaks detected will cost more for each
            % time instance - need to scale for low speed regions)
            % Smoothness of the curve will decrease the cost
            peak_loc = obj.peak_loc_(obj.turn_index_:end);
            valley_loc = obj.valley_loc_(obj.turn_index_:end);
            peak_val = obj.peak_val_(obj.turn_index_:end);
            valley_val = obj.valley_val_(obj.turn_index_:end);
            peak_valley_val = obj.peak_valley_val_(obj.turn_index_:end);
            
            matrix = obj.similarity_matrix_(:,obj.turn_index_:end);
            [tracked_peaks,segment_fraction] = DP_peak_tracking(matrix,peak_loc,obj.motion_index_,obj.turn_index_,obj.peak_significance_,obj.debug_);
            
            % If a sudden deviation in trace, ignore those parts
            datalength = length(tracked_peaks);
            for i = 1:datalength
                if (~isnan(tracked_peaks(i)) && tracked_peaks(i)~=0)
                    prev_loc = tracked_peaks(i);
                    break;
                end
            end
            for j = i+1:datalength
                if tracked_peaks(j)-prev_loc>50
                    tracked_peaks(j) = nan;
                else
                    prev_loc = tracked_peaks(j);
                end
            end
            obj.tracked_peak_trace_backward_ = tracked_peaks;  
        end
        
        function [potential_peaks,potential_peaks_significances,potential_valleys] = peakAnalysis(obj,signal)
            N = length(signal);    % length of the signal
            L = obj.peak_detection_window_;
            
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
            % Extract the peaks between two valleys
            potential_peaks = [];
            potential_peaks_significances = [];
            potential_valleys = [];
            for i = 2:length(peaks)-1
                if (peaks(1,i-1)<0 && peaks(1,i+1)<0 && peaks(1,i)>0)
                    potential_peaks = [potential_peaks; [abs(peaks(1,i)),peaks(2,i)]];
                    potential_peaks_significances = [potential_peaks_significances , (peaks(2,i)-peaks(2,i-1)+peaks(2,i)-peaks(2,i+1))/2];
                    potential_valleys = [potential_valleys ; [abs(peaks(1,i-1)),peaks(2,i-1)]];
                end
            end
            %% Check if the last extremum is a peaks
            if (size(peaks,2)>1)
                if (peaks(1,end)>0 && peaks(1,end-1)<0)
                    potential_peaks = [potential_peaks ; [abs(peaks(1,end)),peaks(2,end)]];
                    potential_peaks_significances = [potential_peaks_significances , (peaks(2,end)-peaks(2,end-1))];
                    potential_valleys = [potential_valleys ; [abs(peaks(1,end-1)),peaks(2,end-1)]];
                end
            end
        end
        
    end
end

