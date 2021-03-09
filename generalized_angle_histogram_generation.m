clc; clear all; close all;
% Code to read output_sf_x.mat and plot the histogram of the angle data.
excel_filename = 'D:\WiCode\conditional_probability_gesture/prior_probabilities.xlsx';


%% Parameters list
% angle classification
characters_list{1} = {'D','P','T','Y','Z'};
characters_list{2} = {'A','F','J','M','O'};
characters_list{3} = {'B','E','H','K','Q'};

epsilon_radius = 0.25;
true_label_list = [];
counter = 0;

area_ratio_list_acute = [];
peaks_detected_length_list_acute = [];

area_ratio_list_zero = [];
peaks_detected_length_list_zero = [];


direc = dir(fullfile('D:\WiCode\10212020','output_sf_3.mat'));
for ff = 1:size(direc,1)
    feature_file = strcat(direc(ff).folder,'/',direc(ff).name);
    output = load(feature_file);
    output = output.output;
    num_samples = size(output.gesture_data,2);
    for ff = 1:num_samples
        if (isempty(output.gesture_data{ff}))
            break;
        end
        filename = output.gesture_data{ff}.filename;
        groundtruth_character = extractBefore(filename,'shape');
        num_segments = output.gesture_data{ff}.num_segments;
        num_angles = num_segments - 1;
        
        excel_sheetname = strcat(num2str(num_segments),'segment');
        [num,txt,~] = xlsread(excel_filename,excel_sheetname);
        prior_angles = num;
        
        
        comparison_result = strcmp(characters_list{num_segments-2},groundtruth_character);
        true_label = find(comparison_result==1,1,'first');
        if (isempty(true_label))
            disp(['Cannot find true label!']);
            continue;
        else
            true_character = characters_list{num_segments-2}(true_label);
            true_label_list  = [true_label_list true_label];
            disp(['True label: ',true_character{1}]);
        end
        
        
        if (isfield(output.gesture_data{ff},'angle_data'))
            angle_data = output.gesture_data{ff}.angle_data;
            for a = 1:num_angles
                if (~isempty(angle_data{a}))
                    angle_class = num(true_label,a);
                    if angle_class == 1
                        area_ratio_list_zero = [area_ratio_list_zero angle_data{a}.area_ratio];
                        peaks_detected_length_list_zero = [peaks_detected_length_list_zero angle_data{a}.peaks_length];
                    else
                        area_ratio_list_acute = [area_ratio_list_acute angle_data{a}.area_ratio];
                        peaks_detected_length_list_acute = [peaks_detected_length_list_acute angle_data{a}.peaks_length];
                    end                   
                end
            end
        end
    end
end

area_ratio_list_zero(peaks_detected_length_list_zero<0.7) = [];
peaks_detected_length_list_zero(peaks_detected_length_list_zero<0.7) = [];

area_ratio_list_acute(peaks_detected_length_list_acute<0.7) = [];
peaks_detected_length_list_acute(peaks_detected_length_list_acute<0.7) = [];

peaks_detected_length_list_zero(area_ratio_list_zero>1) = [];
area_ratio_list_zero(area_ratio_list_zero>1) = [];

peaks_detected_length_list_acute(area_ratio_list_acute>1) = [];
area_ratio_list_acute(area_ratio_list_acute>1) = [];


min_datalength = min(length(area_ratio_list_zero),length(area_ratio_list_acute));

area_ratio_list_zero = area_ratio_list_zero(1:min_datalength);
peaks_detected_length_list_zero = peaks_detected_length_list_zero(1:min_datalength);
area_ratio_list_acute = area_ratio_list_acute(1:min_datalength);
peaks_detected_length_list_acute = peaks_detected_length_list_acute(1:min_datalength);

area_ratio_list_zero_bak = area_ratio_list_zero;
area_ratio_list_acute_bak = area_ratio_list_acute;

figure;
hzero = histogram(area_ratio_list_zero,50,'Normalization','pdf','BinWidth',0.01); hold on;grid on;
hacute = histogram(area_ratio_list_acute,50,'Normalization','pdf','BinWidth',0.01); hold on;grid on;
xlim([0,1]);
legend('\theta\approx0^\circ','\theta\approx45^\circ');
ax = gca;
ax.FontSize = 14;
xlabel('f');
ylabel('Frequency');


true_labels = [zeros(1,length(area_ratio_list_zero_bak)) ones(1,length(area_ratio_list_acute_bak))];
%Identify Classification threshold
threshold_list = 0:0.0005:0.8;
class1_correct = zeros(1,length(threshold_list));
class2_correct = zeros(1,length(threshold_list));

for th = 1:length(threshold_list)
    threshold = threshold_list(th);
    feature_vector  = [area_ratio_list_zero_bak , area_ratio_list_acute_bak];
    predicted_labels = feature_vector>threshold;
    C = confusionmat(true_labels,double(predicted_labels));
    class1_correct(th) = C(1,1);
    class2_correct(th) = C(2,2);
end

total_accuracy = class1_correct +  class2_correct;
[~,max_ind] = max(total_accuracy);
threshold = threshold_list(max_ind);
disp(['Threshold: ',num2str(threshold)]);

% Angle classification accuracy for the 2 angles
feature_vector  = [area_ratio_list_zero_bak , area_ratio_list_acute_bak];
predicted_labels = feature_vector>threshold;
true_labels = true_labels + 1;
predicted_labels = predicted_labels + 1;
class_labels = {'zero','acute'};
predicted_class = class_labels(predicted_labels);
true_class = class_labels(true_labels);
figure;
cm = confusionchart(true_class, predicted_class, 'Normalization','row-normalized');
classification_accuracy = (cm.NormalizedValues(1,1) +  cm.NormalizedValues(2,2))/2;
disp(['Classification accuracy: ' , num2str(classification_accuracy)]);
ax = gca;
ax.FontSize = 20;
sortClasses(cm,["zero","acute"])

x = 0:0.0005:1;
overall_min = 1;
for s1 = 0.1:0.001:0.3
    for s2 = 0.1:0.001:0.6
        pd1 = makedist('HalfNormal','mu',0,'sigma',s1);
        pd2 = makedist('Normal','mu',1,'sigma',s2);
        
        pdf1 = pdf(pd1,x);
        pdf2 = pdf(pd2,x);
        % Find intersection point
        ydiff = abs(pdf1-2*pdf2);
        [min_val,min_ind] =  min(ydiff);
        if (abs(x(min_ind)-threshold))<overall_min
            overall_min = abs(x(min_ind)-threshold);
            min_s1 = s1;
            min_s2 = s2;
        end
    end
end

pd1 = makedist('HalfNormal','mu',0,'sigma',min_s1);
pd2 = makedist('Normal','mu',1,'sigma',min_s2);
pdf1 = pdf(pd1,x);
pdf2 = pdf(pd2,x);
figure;
plot(x,pdf1,'r','LineWidth',2);
hold on;
plot(x,2*pdf2,'b','LineWidth',2);
hold on;grid on;

factor = pdf1+ 2*pdf2;
normalized_pdf1 = pdf1./factor;
normalized_pdf2 = pdf2./factor;
figure;
plot(x,normalized_pdf1,'r','LineWidth',2);
hold on;
plot(x,2*normalized_pdf2,'b','LineWidth',2);
hold on;grid on;

% s = [min_s1,min_s2];
% save('results/angle_pdf_parameters_sf_3.mat','s');