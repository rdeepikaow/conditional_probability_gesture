clc; clear all; close all;
direc = dir(fullfile('E:\deepika\WiCode\angle_classification_entire_dataset\zero_angle\*','*_tracked.png'));
for ff = 168:size(direc,1)
    filename = strcat(direc(ff).folder,'/',direc(ff).name);
    A = imread(filename);
    image(A);
    decision = input('Does this look correct? : ');
    if (~isempty(decision))
        if ~exist([direc(ff).folder,'/faulty'],'dir')
            mkdir([direc(ff).folder,'/faulty']);
        end
         % move files to faulty folder
         extractDate = extractAfter(direc(ff).folder,'\zero_angle\');
         filename1 = filename;
         filename1_dest = strrep(filename1,extractDate,strcat(extractDate,'/faulty'));
         
         filename2_temp = extractBefore(filename,'-phase-comp.mat');
         filename2 = strrep(filename2_temp,'-','_');
         filename2_dest = strrep(filename2,extractDate,strcat(extractDate,'/faulty'));
         
         filename3 = strcat(filename2,'_phase_comp.mat');
         filename3_dest = strrep(filename3,extractDate,strcat(extractDate,'/faulty'));
         
         movefile(filename1,filename1_dest);
         movefile(filename2,filename2_dest);
         movefile(filename3,filename3_dest);
    end
end
