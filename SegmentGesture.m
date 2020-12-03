classdef SegmentGesture
    % Class with functions to partition the different line segments in a
    % gesture.
    
    properties
        segments_start_;
        segments_stop_;
        ccf_similarity_decay_th_;
        ccf_similarity_decay_smooth_win_;
        num_segments_;
        sounding_rate_set_;
        local_peak_win_;
        motion_index_;
        motion_index_smoothed_;
        debug_;
        min_segment_duration_;
        ccf_similarity_decay_step_;
        peaks_;
        smoothed_ms_;
    end
    
    methods
        function obj = SegmentGesture(params)
            obj.segments_start_ = [];
            obj.segments_stop_ = [];
            obj.sounding_rate_set_ = params.sounding_rate_set;
            obj.ccf_similarity_decay_th_ = params.ccf_similarity_decay_th;
            obj.ccf_similarity_decay_smooth_win_ = params.ccf_similarity_decay_smooth_win;
            obj.num_segments_ = 0;
            obj.local_peak_win_ = params.local_peak_win;
            obj.debug_ = params.debug;
            obj.min_segment_duration_ = params.min_segment_duration;
            obj.ccf_similarity_decay_step_ = params.ccf_similarity_decay_step;
            obj.peaks_ = [];
        end
        
        function obj = CCF_segmentation(obj,matrix)
            Nll = size(matrix,1);
            matrix_decay  = cell(1,Nll);
            motion_index = zeros(1,Nll); % Index at which the speed decay crossed th.
            for i = 1:Nll
                indices = 1:min(i-1,Nll-i);
                a = i - indices;
                b = i + indices;
                temp_array = zeros(1,length(a));
                for j = 1:length(a)
                    temp_array(j) = matrix(a(j),b(j));
                end
                matrix_decay{i} = temp_array;
                %                 matrix_decay{i} = hampel(1:length(matrix_decay{i}),matrix_decay{i},3);
                matrix_decay{i} = movmean(matrix_decay{i},obj.ccf_similarity_decay_smooth_win_);
            end
            flag = 0;
            while(flag~=1)
                motion_index = zeros(1,Nll);
                for i = 1:Nll
                    temp_index = find(matrix_decay{i}<obj.ccf_similarity_decay_th_,1);
                    if (~isempty(temp_index))
                        motion_index(i) = temp_index;
                    end
                end
                if nnz(motion_index==1)>0.50*length(motion_index)
                    obj.ccf_similarity_decay_th_ = obj.ccf_similarity_decay_th_ - obj.ccf_similarity_decay_step_;
                else
                    flag = 1;
                    
                end
            end
            obj.motion_index_ = motion_index;
            motion_index_smoothed = smooth(motion_index,0.2,'loess');
            obj.motion_index_smoothed_ = motion_index_smoothed;
            [pks,peaks] = findpeaks(motion_index_smoothed,'MinPeakDistance',obj.min_segment_duration_*obj.sounding_rate_set_,'MinPeakHeight',10);
            
            % Count segments:Check for start and end of entire gesture
            obj.segments_start_ = peaks;
            if mean(motion_index_smoothed(1:peaks(1)))<motion_index_smoothed(peaks(1))/2 && peaks(1)> obj.min_segment_duration_*obj.sounding_rate_set_
                obj.segments_start_(1) = 1;
                obj.segments_start_(2:length(peaks)+1) = peaks;
            end
            if mean(motion_index_smoothed(peaks(end):end))<motion_index_smoothed(peaks(1))/2 && nnz(motion_index(peaks(end):end)==0)>0.99*(length(motion_index_smoothed)-peaks(end))...
                    && length(motion_index_smoothed)-peaks(end)> obj.min_segment_duration_*obj.sounding_rate_set_
                obj.segments_start_ = [obj.segments_start_ length(motion_index_smoothed)];
            end
            obj.segments_stop_ = obj.segments_start_(2:end);
            obj.segments_start_(end) = [];
            obj.num_segments_ = length(obj.segments_start_);
            if(obj.debug_)
                figure;plot(motion_index_smoothed);hold on;grid on;
                plot(obj.segments_start_,motion_index_smoothed(obj.segments_start_),'r*'); hold on;grid on;
                plot(obj.segments_stop_(end),motion_index_smoothed(obj.segments_stop_(end)),'r*'); hold on;grid on;
            end
        end
        
        
        function obj = ms_segmentation(obj,motion_statistics)
            smoothed_ms = smooth(motion_statistics,0.2,'rloess');
            if (obj.debug_)
                figure;
                plot(1:length(motion_statistics),motion_statistics); hold on;grid on;
                plot(smoothed_ms); hold on; grid on;
                xlabel('CSI sample index');
                ylabel('Motion statistics');
                legend('Raw','Smoothed');
                ax = gca; ax.FontSize = 14;
            end
            obj = obj.find_extremums(smoothed_ms);
            %% Identify the top peak
            top_peak_index = find(obj.peaks_(2,:)>0 & obj.peaks_(1,:)>0, 1);
            if (obj.debug_)
                if (~isempty(top_peak_index))
                    disp(['Atleast one gesture segment detected!']);
                else
                    disp(['No segment is detected!']);
                end
            end
            % Gather the peak prominences for each of the peaks
            peak_significances = zeros(1,length(obj.peaks_));
            for p = 2:length(obj.peaks_)-1
                if (obj.peaks_(1,p-1)<0 && obj.peaks_(1,p+1)<0)
                    peak_significances(p) = (abs(obj.peaks_(2,p-1)-obj.peaks_(2,p))+abs(obj.peaks_(2,p+1)-obj.peaks_(2,p)));
                end
            end
            
            highest_peak_significance = peak_significances(top_peak_index);
            segment_peaks = find(peak_significances>highest_peak_significance/3 & obj.peaks_(2,:)>0.01);
            obj.num_segments_ = length(segment_peaks);
            obj.segments_start_ = zeros(1,obj.num_segments_);
            obj.segments_stop_ = zeros(1,obj.num_segments_);
            for i = 1:obj.num_segments_
                obj.segments_start_(i) = abs(obj.peaks_(1,segment_peaks(i)-1));
                obj.segments_stop_(i) = abs(obj.peaks_(1,segment_peaks(i)+1));
            end
            silent_locations = [obj.segments_start_ obj.segments_stop_(end)];
            if (obj.debug_)
                plot(silent_locations,smoothed_ms(silent_locations),'bp','MarkerSize',10); hold on;grid on;
                plot(obj.peaks_(1,segment_peaks),smoothed_ms(obj.peaks_(1,segment_peaks)),'rp','MarkerSize',10); hold on;grid on;
            end
            obj.smoothed_ms_ = smoothed_ms;
        end
        
        
        function obj = find_extremums(obj,signal)
            %% Function detects local peaks and returns the peak information for the direction function
            N = length(signal);    % length of the signal
            L = 20;
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
            obj.peaks_ = peaks;
        end
    end
end

