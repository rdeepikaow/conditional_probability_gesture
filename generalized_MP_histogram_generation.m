clc; clear all; close all;
% Code to read output_sf_x.mat and plot the histogram of the matching point
% values.
excel_filename = 'D:\WiCode\conditional_probability_gesture/prior_probabilities.xlsx';


%% Parameters list
% angle classification
characters_list{1} = {'D','P','T','Y','Z'};
characters_list{2} = {'A','F','J','M','O'};
characters_list{3} = {'B','E','H','K','Q'};

epsilon_radius = 0.25;
true_label_list = [];
counter = 0;
min_TRRS_MP = [];
max_TRRS_MP = [];
min_TRRS_no_MP = [];
max_TRRS_no_MP = [];
direc = dir(fullfile('D:\WiCode\02012021_LOS_NLOS\NLOS','output_sf_3.mat'));
for d = 1:size(direc,1)
    feature_file = strcat(direc(d).folder,'/',direc(d).name);
    output = load(feature_file);
    output = output.output;
    num_samples = size(output.gesture_data,2);
    for ff = 1:num_samples
        if (isempty(output.gesture_data{ff}))
            continue;
        end
        filename = output.gesture_data{ff}.filename;
        groundtruth_character = extractBefore(filename,'shape');
        if (~isfield(output.gesture_data{ff},'num_segments'))
            continue;
        end            
        num_segments = output.gesture_data{ff}.num_segments;
        if (num_segments<3)
            continue;
        end            
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
        
        
        if (output.include_MP_flag && isfield(output.gesture_data{ff},'matching_point_data'))
            MP_data = output.gesture_data{ff}.matching_point_data;
            num_MP = size(MP_data,2);
            for m = 1:num_MP
                if (~isempty(MP_data{m}))
                    text = txt(1+true_label,1+num_angles+m);
                    class_MP1_raw = extractBetween(text{1},'(',',');
                    class_MP1 = str2num(class_MP1_raw{1});
                    class_MP2_raw = extractBetween(text{1},',',')');
                    class_MP2 = str2num(class_MP2_raw{1});
                    fraction_one = MP_data{m}.fraction_one;
                    fraction_three = MP_data{m}.fraction_three;
                    if (isnan(class_MP1))
                        min_TRRS_no_MP = [min_TRRS_no_MP MP_data{m}.min_TRRS_current];
                        max_TRRS_no_MP = [max_TRRS_no_MP MP_data{m}.max_TRRS];
                    else
                        if (fraction_one<class_MP1 + epsilon_radius &&fraction_one>class_MP1 - epsilon_radius && ...
                                fraction_three<class_MP2 + epsilon_radius && fraction_three>class_MP2 - epsilon_radius)
                            min_TRRS_MP = [min_TRRS_MP MP_data{m}.min_TRRS_current];
                            max_TRRS_MP = [max_TRRS_MP MP_data{m}.max_TRRS];
                        else
                            continue;
                        end
                    end
                end
            end
        end
    end
end
no_MP_feature = (max_TRRS_no_MP - min_TRRS_no_MP)./(1-min_TRRS_no_MP);
MP_feature = (max_TRRS_MP - min_TRRS_MP)./(1-min_TRRS_MP);

datalength = min(length(no_MP_feature),length(MP_feature));

no_MP_feature = no_MP_feature(1:datalength);
MP_feature = MP_feature(1:datalength);


figure;
plot(no_MP_feature); hold on;grid on;
plot(MP_feature); hold on;grid on;
legend('No matching point','matching point');
ax = gca;
ax.FontSize = 14;
ylim([0 1]);

figure;
histogram(no_MP_feature,'NumBins',25,'Normalization','probability'); hold on;grid on;
histogram(MP_feature,'NumBins',25,'Normalization','probability'); hold on;grid on;
ax = gca;
ax.FontSize = 14;
legend('No intersection point','With intersection point');

threshold_list = 0:0.0001:1;
correctly_classified = zeros(1,length(threshold_list));
for th = 1:length(threshold_list)
    true_labels = [zeros(1,datalength) ones(1,datalength)];
    feature_list = [no_MP_feature MP_feature];
    classified_list = feature_list > threshold_list(th);
    correctly_classified(th) = nnz(find(true_labels-classified_list==0));
end

[max_val,max_ind] = max(correctly_classified);
threshold = threshold_list(max_ind);
disp(['Threshold: ', num2str(threshold)]);
disp(['Accuracy: ', num2str(max_val/(2*datalength))]);

%% Construct a half normal density for matching point class
% Ideal value = 1;

sigma_list = 0:0.0001:0.8;
difference = zeros(1,length(sigma_list));
for s = 1:length(sigma_list)
    pd1 = makedist('Normal','mu',1,'sigma',sigma_list(s));
    pdf1 = pdf(pd1,threshold);
    pdf1max = pdf(pd1,1);
    difference(s) = abs(pdf1/pdf1max-0.5);
end

[min_val,min_ind] = min(difference);
sigma = sigma_list(min_ind);
pd1 = makedist('Normal','mu',1,'sigma',sigma);

x = 0:0.00001:1;
pdf1 = pdf(pd1,x);
figure;
plot(x,pdf1./max(pdf1),'r','LineWidth',2);
hold on;grid on;
plot(x, 1-pdf1./max(pdf1),'b','LineWidth',2);
hold on;grid on;
ylabel('Probability');
ax = gca;
ax.FontSize = 14;
legend('Intersection point','No intersection point');

%     s = sigma;
%     save('results/matching_point_pdf_parameters_sf_3.mat','s');


