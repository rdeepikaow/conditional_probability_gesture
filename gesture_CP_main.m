%% Author: Sai Deepika Regani
% Date: October 29th, 2020
% Main script for gesture classification using conditional probability
% approach.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
debug = 1;
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

direc = dir(fullfile('D:\WiCode\*10172020*\','*Yshape_loc6_set3_NLOS_3*.mat_phase_comp.mat'));
accuracy_total = 0;
accuracy_correct = 0;
predicted_characters_indices = [];
groundTruth_characters_indices = [];
character_list = {'D','P','T','X','Y','Z'};
for ff = 1:size(direc,1)
    filename = strcat(direc(ff).folder,'/',direc(ff).name);
    matfilename = filename;
    if contains(filename,'phase_comp.mat')
        matfilename = strrep(filename,'_phase_comp.mat','');
    end
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
    if (num_segments~=3)
        continue;
    end
    groundTruth_characters_indices = [groundTruth_characters_indices find(strcmp(extractedGT{1},character_list),1)];
    
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
    for a = angle_indices
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
        
        if (contains(filename,'phase_comp.mat'))
            y_upper_limit = 0.95;
        else
            y_upper_limit = 0.98;
        end
        
        if (debug)
            figure;
            imagesc(smatrix(start_index:stop_index,start_index:stop_index)); colorbar; hold on;grid on;caxis([y_upper_limit 1]);
            plot((1:length(tracked_peaks_backward))+turn_index-1,tracked_peaks_backward,'b*'); hold on;grid on;
        end
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
            plot(SV); hold on;grid on;ylim([y_upper_limit 1]);
            title('Sample to Valley');
            subplot(3,1,2);
            plot(SP); hold on;grid on;ylim([y_upper_limit 1]);
            title('Sample to Peak');
            subplot(3,1,3);
            plot(PV); hold on;grid on;ylim([y_upper_limit 1]);
            title('Peak to Valley');
            
            figure;
            plot(SV); hold on;grid on;ylim([y_upper_limit 1]);
            plot(SP); hold on;grid on;ylim([y_upper_limit 1]);
            plot(PV); hold on;grid on;ylim([y_upper_limit 1]);
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
        if (debug)
            figure;
            plot(SV); hold on;grid on;
            plot(SP); hold on;grid on;
            plot(PV); hold on;grid on;
            legend('SV','SP','PV');
            ax = gca;
            ax.FontSize = 12;
        end
        
        upper_bound = (SP(find(SP>0.9,1,'first')));
        difference_PV = upper_bound-PV(PV~=0);
        difference_PV(difference_PV<0) = 0;
        difference_SP = sum(upper_bound-SP(SP~=0));
        difference_SP(difference_SP<0) = 0;
        area_under_PV = sum(difference_PV);
        area_under_SP = sum(difference_SP);
        fraction_area = area_under_SP/area_under_PV;
        if (fraction_area==0)
            fraction_area = 2;% Any value greater than 1.
        end
        peaks_detected_fraction = nnz(PV)/(stop_index-start_index-turn_index);
        if (debug)
            disp(['a_f: ', num2str(fraction_area)]);
            disp(['rho_f: ',num2str(peaks_detected_fraction)]);
        end
        P_angle_zero = (1-min(1,fraction_area))*peaks_detected_fraction;
        P_angle_acute = min(1,fraction_area);
        %         P_angle_obtuse = (1-peaks_detected_fraction)*min(1,fraction_area);
        angle_probabilities(1,a) = P_angle_zero;
        angle_probabilities(2,a) = P_angle_acute;
        %         angle_probabilities(3,a) = P_angle_obtuse;
        angle_probabilities(:,a) = angle_probabilities(:,a)./sum(angle_probabilities(:,a));
        [max_probability,max_ind] = max([P_angle_zero,P_angle_acute]);
        if (debug)
            disp(['P(a=0) :',num2str(angle_probabilities(1,a))]);
            disp(['P(a=acute) :',num2str(angle_probabilities(2,a))]);
            %             disp(['P(a=obtuse) :',num2str(angle_probabilities(3,a))]);
        end
        if (debug)
            switch max_ind
                case 1
                    disp('zero degree turn')
                case 2
                    disp('acute degree')
                case 3
                    disp('obtuse degree')
            end
        end
    end
    
    %% Matching point determination
    
    switch num_segments
        case 3
            matching_points_fractions = nan(1,2);
        case 4
            matching_points_fractions = nan(3,2);
        case 5
            matching_points_fractions = nan(6,2);
    end
    if (num_segments>2)
        counter = 0;
        for i = 1:num_segments-2
            for j = i+2:num_segments
                counter = counter + 1;
                matching_point_matrix = smatrix(segments_start(i):segments_stop(i),segments_start(j):segments_stop(j));
                if (debug)
                    figure; imagesc(matching_point_matrix);colorbar; caxis([y_upper_limit 1]);
                end
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
                peaks_row = sort(peaks_row(1,:),'descend');
                peaks_col = sort(peaks_col(1,:),'descend');
                
                matching_point_estimate = 0;
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
                matching_points_fractions(counter,1) = segment_one_fraction;
                matching_points_fractions(counter,2) = segment_three_fraction;
            end
        end
    end
    if (debug)
        disp(['Matching points: ',num2str(matching_points_fractions)]);
    end
    [character,pscore,posterior_CP_matching] = gesture_classification_CP(angle_probabilities,matching_points_fractions);
    predicted_characters_indices = [predicted_characters_indices find(strcmp(character_list,character),1,'first')];
    
    data = [];
    data.angle_probabilities = angle_probabilities;
    data.matching_point_fractions = matching_points_fractions;
    data.matching_point_probabilities = posterior_CP_matching;
    data.pscore = pscore;
    data.characters = character_list;
    savefilename = strrep(filename,'.mat_phase_comp.mat','feature.mat');
    save(savefilename,'data');
    disp(['<strong>The gesture shape is: ',character,'</strong>!']);
    close all;
end
groundTruth_characters = character_list(groundTruth_characters_indices);
predicted_characters = character_list(predicted_characters_indices);
figure;
cm = confusionchart(groundTruth_characters,predicted_characters);
figure;
cm_indices = confusionchart(groundTruth_characters_indices,predicted_characters_indices,...
    'Normalization','row-normalized');


