%% Author: Sai Deepika Regani
% Date: June 16th, 2020
% Main script for reconstructing gesture using 5GHz WiFi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; %close all;
debug = 0;
params_gesture_construction.sf = 4; % Sampling factor
params_gesture_construction.display_min = 0.95; % caxis min for display
params_gesture_construction.outlier_removal_offset = 1; % lag in samples outlier removal
params_gesture_construction.outlier_removal_th = 0.97; % Similarity th for outlier removal
params_gesture_construction.debug = 1;
params_gesture_construction.crop_index = -1; % Index at which CSI time series is cropped
params_gesture_construction.num_useful_subcarriers = 248; % Applicable for MRC and correlation
params_gesture_construction.similarity_method = 'trrs'; %'trrs','correlation_phaseboost','correlation'

params_angle_estimation.peak_detection_window = 100;
params_angle_estimation.peak_tracking_max_queue = 50;
params_angle_estimation.peak_tracking_max_jump = 100;
params_angle_estimation.debug = 0;
params_angle_estimation.sf = params_gesture_construction.sf;

if (strcmp(params_gesture_construction.similarity_method,'trrs'))
    params_gesture_segmentation.ccf_similarity_decay_th = 0.985;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.002;
else
    params_gesture_segmentation.ccf_similarity_decay_th = 0.6;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.1;
end

params_gesture_segmentation.ccf_similarity_decay_smooth_win = 100;
params_gesture_segmentation.sounding_rate_set = 3000; %Hz
params_angle_estimation.sounding_rate_set = params_gesture_segmentation.sounding_rate_set;
params_gesture_segmentation.min_segment_duration = 0.3; %sec
params_gesture_segmentation.local_peak_win = 96;
params_gesture_segmentation.debug = 0;

direc = dir(fullfile('C:\Users\origin\Desktop\Gesture5GHzDemo\GestureWNC\10172020_50samples_3segments\','Bshape_loc6_test_NLOS_2*.mat'));
gesture_shape_accuracy = 0;
for ff = 1
    %     tic;
    matching_point_Tshape_flag = 0;
    matching_point_Yshape_flag = 0;
    gesture_length_Yshape_flag = 0;
    gesture_length_Tshape_flag = 0;
    filename = strcat(direc(ff).folder,'/',direc(ff).name);
    matfilename = filename;
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
    %     disp(['Number of segments: ',num2str(num_segments)]);
    if (num_segments~=3)
        continue;
    end
    
    smoothed_ms = segment_gesture.smoothed_ms_;
    %% Turn angle estimation
    params_angle_estimation.matfilename = matfilename;
    params_angle_estimation.similarity_matrix = gesture_construction.similarity_matrix_;
    params_angle_estimation.similarity_matrix_smoothed = smatrix;
    turn_angle = TurnAngleEstimation(params_angle_estimation);
    peak_value = cell(1,num_segments-1);
    peak_valley_value = cell(1,num_segments-1);
    final_turn_angles = zeros(1,num_segments-1);
    
    %% Matching point determination
    matching_point_matrix = smatrix(segments_start(1):segments_stop(1),segments_start(3):segments_stop(3));
    figure; imagesc(matching_point_matrix);colorbar; caxis([0.98 1]);
    [max_val,max_loc]= max(matching_point_matrix(:));
    [max_loc_row,max_loc_col]=ind2sub(size(matching_point_matrix),max_loc);
    % Check if it a real peak
    row_vector = matching_point_matrix(max_loc_row,1:end);
    row_vector = hampel(1:length(row_vector),row_vector,5);
    row_vector_smooth = smooth(row_vector,0.2,'rloess');
    col_vector = matching_point_matrix(1:end,max_loc_col);
    col_vector = hampel(1:length(col_vector),col_vector,5);
    col_vector_smooth = smooth(col_vector,0.2,'rloess');
    peaks_row = peaks_matching_point(row_vector_smooth);
    peaks_col = peaks_matching_point(col_vector_smooth);
    
    if (isempty(peaks_row) || isempty(peaks_col))
        continue;
    end
    peaks_row = sort(peaks_row(2,:),'descend');
    peaks_col = sort(peaks_col(2,:),'descend');
    
    matching_point_estimate = 0;
    if peaks_col(1)>0.992 && peaks_row(1)>0.992
        %     if peaks_row(1)-peaks_row(2)>=0.0004 && peaks_col(1)-peaks_col(2)>=0.0004 % If peak is significantly higher
        if (debug)
            figure; subplot(2,1,1);
            plot(row_vector_smooth); hold on;grid on;
            subplot(2,1,2);
            plot(col_vector_smooth); hold on;grid on;
        end
        segment_one_ms = smoothed_ms(segments_start(1):segments_stop(1));
        segment_three_ms = smoothed_ms(segments_start(3):segments_stop(3));
        segment_one_ms(segment_one_ms<0) = 0;
        segment_three_ms(segment_three_ms<0)= 0 ;
        segment_one_fraction = sum(segment_one_ms(1:max_loc_row))/sum(segment_one_ms);
        segment_three_fraction = sum(segment_three_ms(1:max_loc_col))/sum(segment_three_ms);
        if (debug)
            disp(['Segment 1: ',num2str(segment_one_fraction)]);
            disp(['Segment 2: ',num2str(segment_three_fraction)]);
        end
        % P shape
        matching_point_estimate = 0;
        if (segment_one_fraction>0.15 && segment_one_fraction<0.75 && segment_three_fraction>0.75)
            disp(['Matching point module predicts P shape']);
            matching_point_estimate = 1;
        elseif (segment_one_fraction<0.25 && segment_three_fraction>0.75) % D shape
            disp(['Matching point module predicts D shape']);
            matching_point_estimate = 2;
        elseif (segment_one_fraction>0.2 && segment_one_fraction<0.8 && ...
                segment_three_fraction>0.2 && segment_three_fraction<0.8) % X shape
            disp(['Matching point module predicts X shape']);
            matching_point_estimate = 3;
        elseif (segment_one_fraction>0 && segment_one_fraction<0.7 && ...
                segment_three_fraction<0.25) % T shape
            matching_point_Tshape_flag = 1;
            disp(['Matching point module predicts T shape']);
        elseif (segment_one_fraction>0.75 && segment_three_fraction>0.3) % Y shape
            matching_point_Yshape_flag = 1;
            disp(['Matching point module predicts Y shape']);
        end
    else
        disp(['Matching point not detected']);
    end
    %     close all;
    % end
    %%% Gesture length features to confirm 180 deg turns %%%%
    smoothed_ms(smoothed_ms<0) = 0;
    motion_statistics(motion_statistics<0) =0 ;
    segment1_length = sum(motion_statistics(segments_start(1):segments_stop(1)));
    segment2_length = sum(motion_statistics(segments_start(2):segments_stop(2)));
    segment3_length = sum(motion_statistics(segments_start(3):segments_stop(3)));
    if (segment2_length>0.25*segment1_length && segment2_length<0.75*segment1_length)
        %     if (segment2_length>0.25*segment1_length && segment2_length<0.75*segment1_length)
        disp(['Gesture length predicts T shape']);
        gesture_length_Tshape_flag = 1;
    end
    if (segment2_length>0.25*segment3_length && segment2_length<0.8*segment3_length)
        %     if (1.25*segment2_length<segment3_length)
        disp(['Gesture length predicts Y shape']);
        gesture_length_Yshape_flag = 1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    turn_angle_estimate = ones(1,num_segments-1);%0, 1=> acute, 2=> obtuse
    if (contains(filename,'Tshape')==1)
        test_set_a = [1];
    elseif (contains(filename,'Yshape')==1)
        test_set_a = [2];
    else
        test_set_a = [1,2];
    end
    
    for a = test_set_a
        start_index = segments_start(a);
        stop_index = segments_stop(a+1)-1;
        turn_index = segments_stop(a)-segments_start(a)+1; % relative
        motion_index = segment_gesture.motion_index_smoothed_;
        cummulative_ms = (smoothed_ms(start_index:stop_index));
        %turn_angle = turn_angle.quick_turn_angle_estimation(start_index,stop_index,turn_index,cummulative_ms);
        
        turn_angle = turn_angle.peak_valley_similarity_backward(start_index,stop_index,turn_index,cummulative_ms);
        tracked_peaks_backward = turn_angle.tracked_peak_trace_backward_;
        
        [~,outliers] = hampel(1:length(tracked_peaks_backward),tracked_peaks_backward,30);
        tracked_peaks_backward(outliers) = nan;
        
        figure;
        imagesc(smatrix(start_index:stop_index,start_index:stop_index)); colorbar; hold on;grid on;caxis([0.99 1]);
        plot((1:length(tracked_peaks_backward))+turn_index-1,tracked_peaks_backward,'b*'); hold on;grid on;
        
        %%%%%%%%%%%%%%%% Features for turn angle classification %%%%%%%%%%%%%%%%%%%
        % SV -> Sample to Valley
        % SP -> Sample to Peak
        % PV -> Peak to Valley
        SV = zeros(1,length(start_index+turn_index-1:stop_index));
        SP = zeros(1,length(start_index+turn_index-1:stop_index));
        PV = zeros(1,length(start_index+turn_index-1:stop_index));
        matrix = smatrix(start_index:stop_index,start_index:stop_index);
        for i = 1:min(length(SV),length(tracked_peaks_backward))
            if (~isnan(tracked_peaks_backward(i)) && tracked_peaks_backward(i)~=0)
                SV(i) = matrix(turn_index+i-1,turn_index);
                SP(i) = matrix(turn_index+i-1,tracked_peaks_backward(i));
                PV(i) = matrix(turn_index,tracked_peaks_backward(i));
            end
        end
        
        if (debug)
            figure;
            subplot(3,1,1);
            plot(SV); hold on;grid on;ylim([0.98 1]);
            title('Sample to Valley');
            subplot(3,1,2);
            plot(SP); hold on;grid on;ylim([0.98 1]);
            title('Sample to Peak');
            subplot(3,1,3);
            plot(PV); hold on;grid on;ylim([0.98 1]);
            title('Peak to Valley');
            
            figure;
            plot(SV); hold on;grid on;ylim([0.98 1]);
            plot(SP); hold on;grid on;ylim([0.98 1]);
            plot(PV); hold on;grid on;ylim([0.98 1]);
            legend('SV','SP','PV');
            ax = gca;
            ax.FontSize = 12;
        end
        
        %% Raw feature processing
        SV = SV(SV~=0);
        SP = SP(SP~=0);
        PV = PV(PV~=0);
        % Remove outliers
        SV = hampel(1:length(SV),SV,10);
        SP = hampel(1:length(SP),SP,10);
        PV = hampel(1:length(PV),PV,10);
        SV = movmean(SV,20);
        SP = movmean(SP,20);
        PV = movmean(PV,20);
        figure;
        plot(SV); hold on;grid on;
        plot(SP); hold on;grid on;
        plot(PV); hold on;grid on;
        legend('SV','SP','PV');
        ax = gca;
        ax.FontSize = 12;
        
        
        upper_bound = (SP(find(SP>0.9,1,'first')));
        difference_SV = upper_bound-SV(SV~=0);
        difference_SV(difference_SV<0) = 0;
        difference_SP = sum(upper_bound-SP(SP~=0));
        difference_SP(difference_SP<0) = 0;
        area_under_SV = sum(difference_SV);
        area_under_SP = sum(difference_SP);
        fraction_area = area_under_SP/area_under_SV;
        disp(['Fraction SV: ', num2str(fraction_area)]);
        peaks_detected_fraction = nnz(SV)/(stop_index-start_index-turn_index);
        if (fraction_area<0.4 && peaks_detected_fraction>0.2)
            disp(['Turn angle is 180 degrees!']);
            gesture_shape_accuracy = gesture_shape_accuracy + 1;
            disp(['accuracy: ', num2str(gesture_shape_accuracy/size(direc,1)*100)]);
            turn_angle_estimate(a) = 0 ;
        end
    end
    
    
    % Analyse features (for 180 vs acute)
    %     windowSize = 30;
    %     featureLength = length(SV);
    %     stepSize = 10;
    %     delta = 1e-4; % PV~=SV testing
    %     delta_var = 2e-7; % PV~=SV testing
    %     PV_SV_counter = 0 ;
    %     PV_SV_counter_th = 2;
    %     for i = 1:stepSize:featureLength-windowSize
    %         % PV~=SV and SP>>PV and PV,SV decrease and SP almost same.
    %         % PV~=SV
    %         SV_win = SV(i:i+windowSize-1);
    %         PV_win = PV(i:i+windowSize-1);
    %         SP_win = SP(i:i+windowSize-1);
    %         x = mean(SV_win-PV_win);
    %         y = var(SV_win-PV_win);
    %
    %         %             if (nnz(PV_win>SV_win)>0.9*windowSize)
    %         %                 PV_SV_counter = PV_SV_counter +1;
    %         %                 if (PV_SV_counter > PV_SV_counter_th)
    %         %             disp(['Turn angle is acute']);
    %         %                     turn_angle_estimate(a) = 1;
    %         %                     break;
    %         %                 end
    %         %             else
    %         if ((abs(mean(SV_win-PV_win))< delta && var(SV_win-PV_win) < delta_var) || nnz(SV_win>PV_win)>0.9*windowSize)
    %             if (nnz(SP_win<PV_win)==0 && nnz(SP_win<SV_win)==0) % SP always > PV and SV.
    %                 PV_SV_counter = PV_SV_counter +1;
    %                 if (PV_SV_counter > PV_SV_counter_th)
    %                     disp(['Turn angle is 180 deg']);
    %                     turn_angle_estimate(a) = 0 ;
    %                     break;
    %                 end
    %             end
    %         end
    %         %             end
    %     end
    disp(['Turn angle estimates: ',num2str(turn_angle_estimate)]);
    %% Final classification
    % Correct T shape estimates
    if nnz([gesture_length_Tshape_flag, matching_point_Tshape_flag,~turn_angle_estimate(1)])>1
        turn_angle_estimate(1)=0;
    else
        if (turn_angle_estimate(1)==0)
            turn_angle_estimate(1) = 1;
        end
    end
    % Correct Y shape estimates
    if nnz([gesture_length_Yshape_flag, matching_point_Yshape_flag,~turn_angle_estimate(2)])>1
        turn_angle_estimate(2)=0;
    else
        if (turn_angle_estimate(2)==0)
            turn_angle_estimate(2) = 1;
        end
    end
    disp(['ff: ',num2str(ff)]);
    if (turn_angle_estimate(1)==0)
        disp(['<strong>T shape</strong>!']);
    elseif (turn_angle_estimate(2)==0)
        disp(['<strong>Y shape</strong>!']);
    elseif matching_point_estimate == 1
        disp(['<strong>P shape</strong>!']);
    elseif matching_point_estimate == 2
        disp(['<strong>D shape</strong>!']);
    elseif matching_point_estimate == 3
        disp(['<strong>X shape</strong>!']);
    else
        disp(['<strong>Z shape</strong>!']);
    end
    %     toc;
    close all;
end
