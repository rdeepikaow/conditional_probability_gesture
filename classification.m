clc; clear all; close all;
% Code to read the gesture features and classify them
excel_filename = 'D:\WiCode\conditional_probability_gesture/prior_probabilities.xlsx';
excel_sheetname = '3segment';
[num,txt,~] = xlsread(excel_filename,excel_sheetname);
prior_angles = num;

%% Parameters list
% angle classification
min_peaks_detected_length = 0.7;
extended_characters_list = {'D','P','T','X','Y','Z','None'};
characters_list = {'D','P','T','X','Y','Z'};
% characters_list = {'A','F','O','M','J'};


epsilon_radius = 0.25;
alpha = 0.6; % Total probability split between angle classification and MP
feature_file = 'D:\WiCode\10192020_additional_data_to_10172020\output_sf_3.mat';
output = load(feature_file);
output = output.output;
data_samples = size(output.gesture_data);
true_label_list = [];
predicted_label_list = [];
for ff = 1:data_samples(2)
    if (isempty(output.gesture_data{ff}))
        continue;
    end
    filename = output.gesture_data{ff}.filename;
    groundtruth_character = extractBefore(filename,'shape');
    comparison_result = strcmp(characters_list,groundtruth_character);
    true_label = find(comparison_result==1,1,'first');
    if (isempty(true_label))
        disp(['Cannot find true label!']);
        continue;
    else
        true_character = characters_list(true_label);
        disp(['True label: ',true_character{1}]);
    end
    if (~isfield(output.gesture_data{ff},'angle_data'))
        continue;
    end
    angle_data = output.gesture_data{ff}.angle_data;
    num_angles = size(angle_data,2);
    if (num_angles~=2)
        continue;
    end
    one_angle_contribution = 1/num_angles;
    
    if (output.include_MP_flag)
        MP_data = output.gesture_data{ff}.matching_point_data;
        num_MP = size(MP_data,2);
        one_MP_contribution = 1/num_MP;
    end
    
    angle_features = zeros(2,num_angles);
    for a = 1:num_angles
        if isempty(angle_data{a})
            break;
        end
        peaks_length = angle_data{a}.peaks_length;
        area_ratio = angle_data{a}.area_ratio;
        if (peaks_length>min_peaks_detected_length)
            angle_features(1,a) = angle_pdf(area_ratio);
            angle_features(2,a) = 1 - angle_features(1,a);
        else
            angle_features(2,a) = 1;
        end
    end
    if isempty(angle_data{a})
        continue;
    end
    
    character_probabilities_angles = zeros(1,length(characters_list));
    for c = 1:length(characters_list)
        for a = 1:size(prior_angles,2)
            character_probabilities_angles(c) = character_probabilities_angles(c) + angle_features(prior_angles(c,a),a).*one_angle_contribution;
        end
    end
    
    if (output.include_MP_flag)
        character_probabilities_MP = zeros(1,length(characters_list));
        for c = 1:length(characters_list)
            for m = 1:num_MP
                text = txt(1+c,1+num_angles+m);
                class_MP1_raw = extractBetween(text{1},'(',',');
                class_MP1 = str2num(class_MP1_raw{1});
                class_MP2_raw = extractBetween(text{1},',',')');
                class_MP2 = str2num(class_MP2_raw{1});
                MP_data = output.gesture_data{ff}.matching_point_data{m};
                if (isempty(MP_data))
                    continue;
                end
                fraction_one = MP_data.fraction_one;
                fraction_three = MP_data.fraction_three;
                if (fraction_one<class_MP1 + epsilon_radius &&fraction_one>class_MP1 - epsilon_radius && ...
                        fraction_three<class_MP2 + epsilon_radius && fraction_three>class_MP2 - epsilon_radius)
                        matching_point_test_fraction = (MP_data.max_TRRS -MP_data.min_TRRS)/(1-MP_data.min_TRRS);
                    current_MP = matching_point_pdf(matching_point_test_fraction); 
                else
                    current_MP = 0; 
                end
                if (isnan(class_MP1) && isnan(class_MP2))
                    matching_point_test_fraction = (MP_data.max_TRRS -MP_data.min_TRRS)/(1-MP_data.min_TRRS);
                    current_MP = 1- matching_point_pdf(matching_point_test_fraction);
                end
                character_probabilities_MP(c) = character_probabilities_MP(c) + current_MP*one_MP_contribution;
            end
        end
    end
    
    % Total probability
    total_character_probabilities = character_probabilities_angles.*alpha + (1-alpha).*character_probabilities_MP;
    disp(['Total probabilities: ', num2str(total_character_probabilities)]);
    [max_val,max_ind] = max(total_character_probabilities);
    if (nnz(total_character_probabilities==total_character_probabilities(max_ind))>1)
        max_ind = length(characters_list)+ 1;  % The "none" class.
    end
    predicted_label_list = [predicted_label_list max_ind];
    true_label_list  = [true_label_list true_label];
    
    predicted_character = extended_characters_list(max_ind);
    disp(['Predicted : ', predicted_character{1}]);
end

% Obtain the confusion matrix
true_label_list(predicted_label_list==7) = [];
predicted_label_list(predicted_label_list==7) = [];
groundTruth_characters = characters_list(true_label_list);
predicted_characters = characters_list(predicted_label_list);
figure;
cm = confusionchart(groundTruth_characters,predicted_characters,'Normalization','row-normalized');
sortClasses(cm,characters_list)
ax = gca;
ax.FontSize = 16;

figure;
cm = confusionchart(groundTruth_characters,predicted_characters);
sortClasses(cm,characters_list)
ax = gca;
ax.FontSize = 16;

% total accuracy
accuracy_matrix = cm.NormalizedValues;
accuracy = mean(diag(accuracy_matrix));
disp(['Mean accuracy: ', num2str(accuracy*100),' %']);

