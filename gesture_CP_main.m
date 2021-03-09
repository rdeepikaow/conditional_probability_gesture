%% Author: Sai Deepika Regani
% Date: October 29th, 2020
% Main script for gesture classification using conditional probability
% approach.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
addpath('C:\Users\origingpu\Desktop\5GHz_Gesture');
debug = 1;
params_gesture_construction.sf = 3; % Sampling factor
params_gesture_construction.display_min = 0.95; % caxis min for display
params_gesture_construction.outlier_removal_offset = 1; % lag in samples outlier removal
params_gesture_construction.outlier_removal_th = 0.92; % Similarity th for outlier removal
params_gesture_construction.debug = 1;
params_gesture_construction.crop_index = -1; % Index at which CSI time series is cropped
params_gesture_construction.num_useful_subcarriers = 248; % Applicable for MRC and correlation
params_gesture_construction.similarity_method = 'trrs'; %'trrs','correlation_phaseboost','correlation'

params_angle_estimation.peak_detection_window = 100;
params_angle_estimation.peak_tracking_max_queue = 50;
params_angle_estimation.peak_tracking_max_jump = 100;
params_angle_estimation.debug = 1;
params_angle_estimation.sf = params_gesture_construction.sf;

if (strcmp(params_gesture_construction.similarity_method,'trrs'))
    params_gesture_segmentation.ccf_similarity_decay_th = 0.985;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.002;
else
    params_gesture_segmentation.ccf_similarity_decay_th = 0.6;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.1;
end

params_gesture_segmentation.ccf_similarity_decay_smooth_win = 100;
params_gesture_segmentation.sounding_rate_set = 1500; %Hz
params_angle_estimation.sounding_rate_set = params_gesture_segmentation.sounding_rate_set;
params_gesture_segmentation.min_segment_duration = 0.3; %sec
params_gesture_segmentation.local_peak_win = 96;
params_gesture_segmentation.debug = 0;

search_string = '*shape*.mat_phase_comp.mat';
direc = dir(fullfile('D:\WiCode\10172020_50samples_3segments\',search_string));
accuracy_total = 0;
accuracy_correct = 0;
predicted_characters_indices = [];
groundTruth_characters_indices = [];
character_list = {'D','P','T','X','Y','Z','None'};

area_ratio_list = [];
peaks_detected_length_list = [];
include_MP_module = 1;
include_angle_module = 1;
save_output = 1;

output = []; % All results from data in this folder
output.search_string = search_string;
output.gesture_data = cell(1,size(direc,1));
output.include_MP_flag = include_MP_module;
output.sf = params_gesture_construction.sf;
for ff = 1:size(direc,1)
    filename = strcat(direc(ff).folder,'/',direc(ff).name);
    matfilename = filename;
    if contains(filename,'phase_comp.mat')
        matfilename = strrep(filename,'_phase_comp.mat','');
    end
    output.gesture_data{ff}.filename = direc(ff).name;
    extractedGT = extractBetween(filename,'/','shape');
    disp(['Processing: ', direc(ff).name]);
    params_gesture_construction.matfilename = matfilename;
    params_gesture_construction.filename = filename;
    gesture_construction = GestureConstruction(params_gesture_construction);
    gesture_construction = gesture_construction.preprocess(filename);
    smatrix = gesture_construction.similarity_matrix_;
    motion_statistics = gesture_construction.motion_statistics_(max(1,params_gesture_construction.sf-1):end);
    %% Gesture segmentation
    segment_gesture = SegmentGesture(params_gesture_segmentation);
    segment_gesture = segment_gesture.ms_segmentation(motion_statistics); % segmentation based on motion statistics
    segments_start = segment_gesture.segments_start_;
    segments_stop = segment_gesture.segments_stop_;
    num_segments = segment_gesture.num_segments_;
    disp(['Number of segments: ',num2str(num_segments)]);
    %
    %     if num_segments~=5
    %         continue;
    %     end
    %
    if (contains(direc(ff).name,'cute'))
        if (num_segments~=2)
            continue;
        end
    end
    if (contains(direc(ff).name,'180'))
        if (num_segments~=2)
            continue;
        end
    end
    if (contains(direc(ff).name,'Ashape') || contains(direc(ff).name,'Jshape'))
        if (num_segments~=4)
            continue;
        end
    end
    if (contains(direc(ff).name,'Xshape') || contains(direc(ff).name,'Pshape') || contains(direc(ff).name,'Zshape'))
        if (num_segments~=3)
            continue;
        end
    end
    if (contains(direc(ff).name,'Tshape') || contains(direc(ff).name,'Yshape'))
        if (num_segments~=3)
            continue;
        end
    end
    
    % groundTruth_characters_indices = [groundTruth_characters_indices find(strcmp(extractedGT{1},character_list),1)];
    output.gesture_data{ff}.num_segments = num_segments;
    accuracy_total = accuracy_total + 1;
    smoothed_ms = segment_gesture.smoothed_ms_;
    %% Turn angle estimation
    params_angle_estimation.matfilename = matfilename;
    params_angle_estimation.similarity_matrix = gesture_construction.similarity_matrix_;
    params_angle_estimation.similarity_matrix_smoothed = smatrix;
    turn_angle = TurnAngleEstimation(params_angle_estimation);
    peak_value = cell(1,num_segments-1);
    peak_valley_value = cell(1,num_segments-1);
    final_turn_angles = zeros(1,num_segments-1);
    
    angle_indices = 1:num_segments-1;
    angle_probabilities = zeros(2,num_segments-1);%0=> zero, 1=> acute, 2=> obtuse
    %% Temporary addition to select angles with 180 deg turn
    
    %     if (contains(direc(ff).name,'Jshape') || contains(direc(ff).name,'Tshape'))
    %         angle_indices = [1];
    %     end
    %     if (contains(direc(ff).name,'Ashape') || contains(direc(ff).name,'Yshape'))
    %         angle_indices = [2];
    %     end
    %     if (contains(direc(ff).name,'Xshape') || contains(direc(ff).name,'Zshape'))
    %         angle_indices = [1,2];
    %     end
    %     if (contains(direc(ff).name,'Pshape'))
    %         angle_indices = [1];
    %     end
    %
    output.gesture_data{ff}.angle_data = cell(1,length(angle_indices));
    if (include_angle_module)
        for a = angle_indices
            segment2_shorter_flag = 0;
            segment1_ms = smoothed_ms(segments_start(a):segments_stop(a));
            segment2_ms = smoothed_ms(segments_start(a+1):segments_stop(a+1));
            segment1_ms(segment1_ms<0) = 0;
            segment2_ms(segment2_ms<0) = 0;
            segment1_length = sum(segment1_ms);
            segment2_length = sum(segment2_ms);
            if segment2_length*1.2<segment1_length % then swap the two segments
                segment2_shorter_flag = 1;
            end
            start_index = segments_start(a);
            stop_index = segments_stop(a+1)-1;
            turn_index = segments_start(a+1)-segments_start(a)+1; % relative
            motion_index = segment_gesture.motion_index_smoothed_;
            cummulative_ms = (smoothed_ms(start_index:stop_index));
            first_segment_stop_index = segments_stop(a)-start_index+1;
            temp_smatrix = smatrix(start_index:stop_index,start_index:stop_index);
            
            turn_angle.similarity_matrix_smoothed_ = temp_smatrix;
            turn_angle = turn_angle.peak_valley_similarity_backward(turn_index,cummulative_ms,first_segment_stop_index);
            tracked_peaks_backward = turn_angle.tracked_peak_trace_backward_;
            
            if ~isempty(tracked_peaks_backward)
                
                [~,outliers] = hampel(1:length(tracked_peaks_backward),tracked_peaks_backward,30);
                tracked_peaks_backward(outliers) = nan;
            end
            
            if (contains(filename,'phase_comp.mat'))
                y_upper_limit = 0.95;
            else
                y_upper_limit = 0.98;
            end
            
            if (debug)
                figure;
                imagesc(temp_smatrix); colorbar; hold on;grid on;caxis([y_upper_limit 1]);
                plot((1:length(tracked_peaks_backward))+turn_index-1,tracked_peaks_backward,'b*'); hold on;grid on;
                title_name = strrep(direc(ff).name,'_','-');
                title(title_name);
                saveas(gca,strcat(direc(ff).folder,'/',title_name,'_tracked_sf_3.png'));
            end
            
            %         output.gesture_data{ff}.matrix = temp_smatrix;
            tracked_peaks_backward = clean_trace(tracked_peaks_backward);
            if (debug)
                figure;
                imagesc(temp_smatrix); colorbar; hold on;grid on;caxis([y_upper_limit 1]);
                plot((1:length(tracked_peaks_backward))+turn_index-1,tracked_peaks_backward,'b*'); hold on;grid on;
                title_name = strrep(direc(ff).name,'_','-');
                title([title_name,' - cleaned']);
                saveas(gca,strcat(direc(ff).folder,'/',title_name,'_tracked.png'));
            end
            %% find the first index
            first_peak_on_segment1 = tracked_peaks_backward(find(~isnan(tracked_peaks_backward),1,'first'));
            first_peak_fraction = 1- sum(smoothed_ms(segments_start(a):segments_start(a)+first_peak_on_segment1))/sum(smoothed_ms(segments_start(a):segments_start(a+1)-1));
            
            %% Actual number of points a peak is detected
            % number of instances between 0.1 and 0.9 ms of first segment
            cummulative_ms_first_segment = cumsum(smoothed_ms(segments_start(a):segments_start(a+1)-1));
            cummulative_ms_first_segment = cummulative_ms_first_segment./(cummulative_ms_first_segment(end));
            total_instances = find(cummulative_ms_first_segment>0.9,1)-find(cummulative_ms_first_segment>0.1,1);
            detected_instances = nnz(~isnan(tracked_peaks_backward));
            fraction_detected = detected_instances/total_instances;
            
            if (first_peak_fraction>0.5 || fraction_detected<0.25)
                area_ratio_list = [area_ratio_list 10];
                peaks_detected_length_list = [peaks_detected_length_list 0];
                output.gesture_data{ff}.angle_data{a}.area_ratio = 10;
                output.gesture_data{ff}.angle_data{a}.peaks_length = 0;
                angle_probabilities(3,a) = 1;
                continue;
            end
            %%%%%%%%%%%%%%%% Features for turn angle classification %%%%%%%%%%%%%%%%%%%
            % ST -> Sample to Valley
            % SP -> Sample to Peak
            % PT -> Peak to Valley
            ST = zeros(1,length(start_index+turn_index-1:stop_index));
            SP = zeros(1,length(start_index+turn_index-1:stop_index));
            PT = zeros(1,length(start_index+turn_index-1:stop_index));
            matrix = smatrix(start_index:stop_index,start_index:stop_index);
            for i = 1:min(length(ST),length(tracked_peaks_backward))
                if (~isnan(tracked_peaks_backward(i)) && tracked_peaks_backward(i)~=0)
                    ST(i) = matrix(turn_index+i-1,turn_index);
                    SP(i) = matrix(turn_index+i-1,tracked_peaks_backward(i));
                    PT(i) = matrix(turn_index,tracked_peaks_backward(i));
                end
            end
            
            if (debug)
                figure;
                subplot(3,1,1);
                plot(ST); hold on;grid on;ylim([y_upper_limit 1]);
                title('Sample to Valley');
                subplot(3,1,2);
                plot(SP); hold on;grid on;ylim([y_upper_limit 1]);
                title('Sample to Peak');
                subplot(3,1,3);
                plot(PT); hold on;grid on;ylim([y_upper_limit 1]);
                title('Peak to Valley');
                
                figure;
                plot(ST); hold on;grid on;ylim([y_upper_limit 1]);
                plot(SP); hold on;grid on;ylim([y_upper_limit 1]);
                plot(PT); hold on;grid on;ylim([y_upper_limit 1]);
                legend('ST','SP','PT');
                ax = gca;
                ax.FontSize = 12;
                
                % figure;
                % plot(turn_angle.peak_significance_(turn_angle.turn_index_:end)); hold on;grid on;
                % title('Peak significance');
            end
            
%             output.gesture_data{ff}.angle_data{a}.min_TRRS = min(ST);
            
            %% Raw feature processing
            ST = hampel(1:length(ST),ST,5);
            SP = hampel(1:length(SP),SP,5);
            PT = hampel(1:length(PT),PT,5);
            ST_first_nz_index = find(ST~=0,1,'first');
            PT_first_nz_index = find(PT~=0,1,'first');
            SP_first_nz_index = find(SP~=0,1,'first');
            
            ST_last_nz_index = find(ST~=0,1,'last');
            PT_last_nz_index = find(PT~=0,1,'last');
            SP_last_nz_index = find(SP~=0,1,'last');
            
            ST = ST(ST~=0);
            SP = SP(SP~=0);
            PT = PT(PT~=0);
            % Remove outliers
            ST = hampel(1:length(ST),ST,10);
            SP = hampel(1:length(SP),SP,10);
            PT = hampel(1:length(PT),PT,10);
            ST = movmean(ST,20);
            SP = movmean(SP,20);
            PT = movmean(PT,20);
            
            min_data_length = min(length(SP),length(PT));
            SP = SP(1:min_data_length);
            PT = PT(1:min_data_length);
            
            % Concentrate only on the decaying part of the bessel function
            check_vector = temp_smatrix(turn_index,turn_index:end);
            check_vector = hampel(1:length(check_vector),check_vector,5);
            min_trrs_offset = min(check_vector);
            stop_index_area_calculation = find(PT<min_trrs_offset,1,'first');
            if (isempty(stop_index_area_calculation))
                stop_index_area_calculation = length(PT);
            end
            
            output.gesture_data{ff}.angle_data{a}.PT  = PT;
            output.gesture_data{ff}.angle_data{a}.SP = SP;
            
            PT = PT(1:stop_index_area_calculation);
            SP = SP(1:stop_index_area_calculation);
            
            
            [min_PT,min_PT_index] = min(PT);
            PT = PT(1:min_PT_index);
            SP = SP(1:min_PT_index);
            
            output.gesture_data{ff}.angle_data{a}.stop_index_area_calculation = min_PT_index;
            
            % ST = ST(1:stop_index_area_calculation);
            % output.gesture_data{ff}.angle_data{a}.ST  = ST;
            
            upper_bound = max(SP(find(SP>0.9)));
            difference_ST = upper_bound-ST(ST~=0);
            difference_ST(difference_ST<0) = 0;
            
            difference_PT = upper_bound-PT(PT~=0);
            difference_PT(difference_PT<0) = 0;
            
            difference_SP = (upper_bound-SP(SP~=0));
            difference_SP(difference_SP<0) = 0;
            
            area_under_ST = sum(difference_ST);
            area_under_SP = sum(difference_SP);
            area_under_PT = sum(difference_PT);
            
            fraction_area = area_under_SP/area_under_PT;
            if (fraction_area==0)
                fraction_area = 2;% Any value greater than 1.
            end
            % Segment length defined with motion statistics
            if (isnan(min(tracked_peaks_backward(:)))) || isempty(ST)
                area_ratio_list = [area_ratio_list 10];
                peaks_detected_length_list = [peaks_detected_length_list 0];
                output.gesture_data{ff}.angle_data{a}.area_ratio = 10;
                output.gesture_data{ff}.angle_data{a}.peaks_length = 0;
                angle_probabilities(3,a) = 1;
                continue;
            end
            peaks_detected_fraction2 = sum(segment2_ms(1:PT_last_nz_index))/segment2_length;
            peaks_detected_fraction1 = 1-sum(segment1_ms(1:min(tracked_peaks_backward(:))))/segment1_length;
            if segment2_shorter_flag
                peaks_detected_fraction = peaks_detected_fraction2;
                segment2_shorted_flag = 0;
            else
                peaks_detected_fraction = peaks_detected_fraction1;
            end
            if (debug)
                figure;
                constant_y = upper_bound.*ones(1,length(SP));
                plot(SP,'b-','LineWidth',2); hold on;grid on;
                plot(PT,'k-','LineWidth',2); hold on;grid on;
                plot(constant_y,'k-'); hold on;grid on;
                x = [1:length(SP) ,length(SP):-1:1];
                yy = [SP',constant_y];
                fill(x,yy,[0.00,0.60,1.00],'FaceAlpha',0.5) ; hold on;grid on;
                yy1 = [PT',constant_y];
                for v = 1:10:length(SP)
                    plot([v,v],[constant_y(1),PT(v)],'k-');hold on;grid on;
                end
                legend('SP','PT');
                xlabel('CSI sample index');
                ylabel('TRRS');
                ax = gca;
                ax.FontSize = 12;
            end
            output.gesture_data{ff}.angle_data{a}.area_ratio = fraction_area;
            output.gesture_data{ff}.angle_data{a}.peaks_length = peaks_detected_fraction;
            
            disp(['area ratio: ', num2str(fraction_area)]);
            disp(['peaks detected length: ',num2str(peaks_detected_fraction)]);
            area_ratio_list = [area_ratio_list fraction_area];
            peaks_detected_length_list = [peaks_detected_length_list peaks_detected_fraction];
        end
    end
    
    %% Matching point determination
    if (include_MP_module)
        switch num_segments
            case 3
                matching_points_fractions = nan(1,2);
            case 4
                matching_points_fractions = nan(3,2);
            case 5
                matching_points_fractions = nan(6,2);
        end
        output.gesture_data{ff}.matching_point_data = cell(1,size(matching_points_fractions,1));
        
        % Create extended start and end points
        extended_start_points = zeros(1,num_segments);
        extended_end_points = zeros(1,num_segments);
        for i = 1:num_segments
            if ((i-1)~=0)
                normalized_length_prev_segment = cumsum(smoothed_ms(segments_start(i-1):segments_stop(i-1)))./sum(smoothed_ms(segments_start(i-1):segments_stop(i-1)));
                extended_start_points(i) = segments_start(i-1)-1+ find(normalized_length_prev_segment>=0.9,1,'first');
            else
                extended_start_points(i) = 1;
            end
            if (i~=num_segments)
                normalized_length_next_segment = cumsum(smoothed_ms(segments_start(i+1):segments_stop(i+1)))./sum(smoothed_ms(segments_start(i+1):segments_stop(i+1)));
                extended_end_points(i) = segments_start(i+1)-1+ find(normalized_length_next_segment>=0.1,1,'first');
            else
                extended_end_points(i) = segments_stop(i);
            end
        end
        if (num_segments>2)
            counter = 0;
            for i = 1:num_segments-2
                for j = i+2:num_segments
                    counter = counter + 1;
                    matching_point_matrix = smatrix(extended_start_points(i):extended_end_points(i),extended_start_points(j):extended_end_points(j));
                    %                     if (debug)
                    %                         figure; imagesc(matching_point_matrix);colorbar;colormap jet;caxis([y_upper_limit 1]);
                    %                     end
                    %% zero out the unneccessary matching region for i,j = i+2
                    % 1st segment - consider only (0-0.75)
                    % 3rd segment - consider only (0.25-1)
                    if (j==i+2)
                        matching_point_matrix_bak = matching_point_matrix;
                        segment_one_ms = smoothed_ms(segments_start(i):segments_stop(i));
                        segment_three_ms = smoothed_ms(segments_start(j):segments_stop(j));
                        segment_one_ms(segment_one_ms<0) = 0;
                        segment_three_ms(segment_three_ms<0)= 0;
                        
                        cumsum_segment_one_ms = cumsum(segment_one_ms)/sum(segment_one_ms);
                        cumsum_segment_three_ms = cumsum(segment_three_ms)/sum(segment_three_ms);
                        
                        crop_match_seg_one_stop = find(cumsum_segment_one_ms>0.75,1,'first');
                        crop_match_seg_three_start = find(cumsum_segment_three_ms>0.25,1,'first');
                        
                        % 3rd segment
                        start_index_third = crop_match_seg_three_start+segments_start(j)-extended_start_points(j);
                        matching_point_matrix(:,1:start_index_third) = 0;
                        
                        % 1st segment
                        stop_index_first = crop_match_seg_one_stop + segments_start(i)-extended_start_points(i);
                        matching_point_matrix(stop_index_first:end,:) = 0;
                    end
                    
                    [max_val,max_loc]= max(matching_point_matrix(:));
                    if (j==i+2)
                        matching_point_matrix = matching_point_matrix_bak;
                    end
                    [max_loc_row, max_loc_col]=ind2sub(size(matching_point_matrix),max_loc);
                    % Check if it a real peak
                    row_vector = matching_point_matrix(max_loc_row,1:end);
                    row_vector = hampel(1:length(row_vector),row_vector,5);
                    row_vector_smooth = smooth(row_vector,0.2,'rloess');
                    col_vector = matching_point_matrix(1:end,max_loc_col);
                    col_vector = hampel(1:length(col_vector),col_vector,5);
                    col_vector_smooth = smooth(col_vector,0.2,'rloess');
                    peaks_row = peaks_matching_point(row_vector_smooth);
                    peaks_col = peaks_matching_point(col_vector_smooth);
                    
                    min_val_check_vector = [col_vector' row_vector'];
                    % min_val_check_vector(min_val_check_vector==0) = 1;
                    
                    min_val_TRRS = min(min_val_check_vector);
                    
                    if (isempty(peaks_row) || isempty(peaks_col))
                        continue;
                    end
                    peaks_row = sort(peaks_row(1,:),'descend');
                    peaks_col = sort(peaks_col(1,:),'descend');
                    
                    matching_point_estimate = 0;
                    if (debug)
                        figure; subplot(2,1,1);
                        plot(row_vector_smooth); hold on;grid on;
                        subplot(2,1,2);
                        plot(col_vector_smooth); hold on;grid on;
                    end
                    
                    smooth_row_max = row_vector_smooth(max_loc_col);
                    smooth_col_max = col_vector_smooth(max_loc_row);
                    
                    max_loc_row = max_loc_row -(segments_start(i)-extended_start_points(i));
                    max_loc_col = max_loc_col -(segments_start(j)-extended_start_points(j));
                    
                    % Modified MP detection - extended peaks
                    peak_sig_th = 0.005;
                    if smooth_row_max-row_vector_smooth(end)<peak_sig_th
                        max_loc_col = length(segment_one_ms);
                    end
                    if smooth_row_max-row_vector_smooth(1)<peak_sig_th
                        max_loc_col = 1;
                    end
                    if smooth_col_max-col_vector_smooth(end)<peak_sig_th
                        max_loc_row = length(segment_three_ms);
                    end
                    if smooth_col_max-col_vector_smooth(1)<peak_sig_th
                        max_loc_row = 1;
                    end
                    
                    %%%%%
                    if (max_loc_row>length(segment_one_ms))
                        segment_one_fraction = 1;
                    else
                        segment_one_fraction = sum(segment_one_ms(1:max_loc_row))/sum(segment_one_ms);
                    end
                    if (max_loc_col>length(segment_three_ms))
                        segment_three_fraction = 1;
                    else
                        segment_three_fraction = sum(segment_three_ms(1:max_loc_col))/sum(segment_three_ms);
                    end
                    matching_points_fractions(counter,1) = segment_one_fraction;
                    matching_points_fractions(counter,2) = segment_three_fraction;
                    output.gesture_data{ff}.matching_point_data{counter}.fraction_one = segment_one_fraction;
                    output.gesture_data{ff}.matching_point_data{counter}.fraction_three = segment_three_fraction;
                    output.gesture_data{ff}.matching_point_data{counter}.fraction_one_index = max_loc_row;
                    output.gesture_data{ff}.matching_point_data{counter}.fraction_three_index = max_loc_col;
                    output.gesture_data{ff}.matching_point_data{counter}.min_TRRS_overall = min_val_TRRS;
                    output.gesture_data{ff}.matching_point_data{counter}.min_TRRS_current = min_val_TRRS;
                    output.gesture_data{ff}.matching_point_data{counter}.max_TRRS = max_val;
                    disp(['Max: ', num2str(max_val),'  Min: ', num2str(min_val_TRRS)]);
                end
            end
        end
    end
    close all;
end

if (save_output)
    savefilename = strcat(direc(ff).folder,'/output_sf_3.mat');
    save(savefilename,'output');
end

% predicted_characters_indices(isnan(predicted_characters_indices)) = 7;
% groundTruth_characters = character_list(groundTruth_characters_indices);
% predicted_characters = character_list(predicted_characters_indices);
% figure;
% cm = confusionchart(groundTruth_characters,predicted_characters);

% figure;
% cm_indices = confusionchart(groundTruth_characters_indices,predicted_characters_indices,...
%     'Normalization','row-normalized');
% figure;
% for i = 1:3
%     plot(area_ratio_list{i}, ones(1,length(peaks_detected_length_list{i})),'o'); hold on;grid on;
%     xlabel('area ratio');
%     ylabel('peaks detected length');
%     ax = gca; ax.FontSize = 14;
% end
%%  Histogram plot
% peaks_detected_length_list(area_ratio_list>1) = [];
% area_ratio_list(area_ratio_list>1) = [];

% area_ratio_list_bak = area_ratio_list;
% peaks_detected_length_list_bak = peaks_detected_length_list;
% area_ratio_list(peaks_detected_length_list_bak<0.9) = [];
% peaks_detected_length_list(peaks_detected_length_list_bak<0.9) = [];
% peaks_detected_length_list(area_ratio_list>3) = [];
% area_ratio_list(area_ratio_list>3) = [];
% figure;
% h = histogram(area_ratio_list,100);
% feature = [];
% feature.peaks_detected_length_list = peaks_detected_length_list;
% feature.area_ratio_list = area_ratio_list;
% save('180deg_features_list_1500Hz_02_24.mat','feature');
