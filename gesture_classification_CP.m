function [character,probability_score,posterior_CP_matching] = gesture_classification_CP(angle_probabilities,matching_points_fractions)
% Code to read prior probabilities from excel sheet
excel_filename = 'prior_probabilities.xlsx';
excel_sheetname = 'AnglePrior';
[num,txt,raw] = xlsread(excel_filename,excel_sheetname);
prior_CP_angles = num;
excel_sheetname = 'MPrior';
[num,txt,raw] = xlsread(excel_filename,excel_sheetname);
prior_CP_matching = num(:,1:2);

% Convert each angle probabilities into two angle probabilities
prior_CP_probabilities_angles = zeros(6,4);
for i = 1:6
    prior_CP_probabilities_angles(i,1) = angle_probabilities(1,1)*angle_probabilities(1,2);
    prior_CP_probabilities_angles(i,2) = angle_probabilities(1,1)*angle_probabilities(2,2);
    prior_CP_probabilities_angles(i,3) = angle_probabilities(2,1)*angle_probabilities(1,2);
    prior_CP_probabilities_angles(i,4) = angle_probabilities(2,1)*angle_probabilities(2,2);    
end

% Convert matching point probabilities 
posterior_CP_matching = zeros(1,6);
for c = 1:6
    posterior_CP_matching(c) = 1- (MP_function(abs(matching_points_fractions(1)-prior_CP_matching(c,1)))+MP_function(abs(matching_points_fractions(2)-prior_CP_matching(c,2))))/2;
end
posterior_CP_matching = posterior_CP_matching./sum(posterior_CP_matching);
probability_score = zeros(1,6);
for c = 1:size(probability_score,2)
    probability_score(c) = sum(prior_CP_probabilities_angles(c,:).*prior_CP_angles(c,:));
    probability_score(c) = probability_score(c).*posterior_CP_matching(c);
end

probability_score = probability_score./sum(probability_score);
disp(['Probability scores: ',num2str(probability_score)]);
[~,max_ind] = max(probability_score);
character_list = {'D','P','T','X','Y','Z'};
character = character_list{max_ind};
% disp(['Character: ',character]);
end


