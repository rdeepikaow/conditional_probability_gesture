function [character,character_probabilities] = gesture_classification_CP(angle_probabilities,matching_points_fractions,max_TRRS,min_TRRS)
% Code to read prior probabilities from excel sheet
excel_filename = 'prior_probabilities.xlsx';
excel_sheetname = '3segment';
[num,txt,~] = xlsread(excel_filename,excel_sheetname);
prior_CP_angles = num;

% modify angle probabilities to add acute and obtuse angles
angle_probabilities(2,:) = angle_probabilities(2,:)+ angle_probabilities(3,:);
angle_probabilities(3,:) = [];
nC = size(num,1);
character_probabilities = zeros(nC,1);
for ii = 1:nC
    character_probabilities(ii) = angle_probabilities(num(ii,1),1)*angle_probabilities(num(ii,2),2);
end

% matching point probabilities
MP_probability = (max_TRRS- min_TRRS)/(1-min_TRRS);
% Matching point indicator functions
MP_character_probability = zeros(1,nC);
for ii = 1:nC
    text = txt(1+ii,4);
    class_MP1 = str2num(text{1}(2));
    class_MP2 = str2num(text{1}(4));
    if (isempty(class_MP1))
        MP_character_probability(ii) = 1- MP_probability;
    else
        if (matching_points_fractions(1)<class_MP1+0.25 && matching_points_fractions(1)>class_MP1-0.25 && ...
                matching_points_fractions(2)<class_MP2+0.25 && matching_points_fractions(2)>class_MP2-0.25)
            MP_character_probability(ii) = MP_probability;
        end
    end
end

% Total character probability
for ii = 1:nC
    character_probabilities(ii) = character_probabilities(ii)*MP_character_probability(ii);
end

character_probabilities = character_probabilities./sum(character_probabilities);

disp(['Probability scores: ',num2str(character_probabilities')]);
[~,max_ind] = max(character_probabilities);
character_list = {'D','P','T','X','Y','Z'};
character = character_list{max_ind};
% disp(['Character: ',character]);
end


