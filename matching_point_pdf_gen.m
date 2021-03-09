clc; clear all; close all;
% Code to read the gesture features and classify them
excel_filename = 'D:\WiCode\conditional_probability_gesture/prior_probabilities.xlsx';
excel_sheetname = '3segment';
[num,txt,~] = xlsread(excel_filename,excel_sheetname);
prior_angles = num;

%% Parameters list
% angle classification
characters_list = {'D','P','T','X','Y','Z','M'};

epsilon_radius = 0.25;
true_label_list = [];
counter = 0;
min_TRRS = [];
max_TRRS = [];
min_TRRS_no_MP = [];
max_TRRS_no_MP = [];
num_angles = 2;
direc = dir(fullfile('D:\WiCode\matching_point_pdf_data/*','output_MP_sf_3.mat'));
ff_list = [2,1,3,4];
for ff = ff_list
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
        comparison_result = strcmp(characters_list,groundtruth_character);
        true_label = find(comparison_result==1,1,'first');
        if (isempty(true_label))
            disp(['Cannot find true label!']);
        else
            true_character = characters_list(true_label);
            true_label_list  = [true_label_list true_label];
            disp(['True label: ',true_character{1}]);
        end
        
        label_comparison = strcmp(true_character{1},characters_list);
        true_label_index = find(label_comparison==1,1,'first');
        
        if (output.include_MP_flag && isfield(output.gesture_data{ff},'matching_point_data'))
            MP_data = output.gesture_data{ff}.matching_point_data;
            num_MP = size(MP_data,2);
            one_MP_contribution = 1/num_MP;
            character_probabilities_MP = zeros(1,length(characters_list));
            for c = true_label_index
                for m = 1:num_MP
                    
                    if (contains(filename,'Zshape') || contains(filename,'Mshape'))
                        if (~isempty(MP_data{m}))
                            min_TRRS_no_MP = [min_TRRS_no_MP MP_data{m}.min_TRRS];
                            max_TRRS_no_MP = [max_TRRS_no_MP MP_data{m}.max_TRRS];
                        end
                    else
                        text = txt(1+c,1+num_angles+m);
                        class_MP1_raw = extractBetween(text{1},'(',',');
                        class_MP1 = str2num(class_MP1_raw{1});
                        class_MP2_raw = extractBetween(text{1},',',')');
                        class_MP2 = str2num(class_MP2_raw{1});
                        if (isempty(MP_data{m}))
                            continue;
                        end
                        fraction_one = MP_data{m}.fraction_one;
                        fraction_three = MP_data{m}.fraction_three;
                        if (fraction_one<class_MP1 + epsilon_radius &&fraction_one>class_MP1 - epsilon_radius && ...
                                fraction_three<class_MP2 + epsilon_radius && fraction_three>class_MP2 - epsilon_radius)
                            counter = counter + 1;
                            min_TRRS(counter) = MP_data{m}.min_TRRS;
                            max_TRRS(counter) = MP_data{m}.max_TRRS;
                        end
                    end
                end
            end
        end
    end
end

no_MP_feature = (max_TRRS_no_MP - min_TRRS_no_MP)./(1-min_TRRS_no_MP);
MP_feature = (max_TRRS - min_TRRS)./(1-min_TRRS);

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

s = sigma;
save('results/matching_point_pdf_parameters_sf_3.mat','s');


