function [probability] = angle_pdf(area_ratio)

filename = 'results/angle_pdf_parameters_sf_3.mat';
data = load(filename);
s = data.s;
pd1 = makedist('HalfNormal','mu',0,'sigma',s(1));
pd2 = makedist('Normal','mu',1,'sigma',s(2));
pdf1 = pdf(pd1,area_ratio);
pdf2 = pdf(pd2,area_ratio);

factor = pdf1+ 2*pdf2;
normalized_pdf1 = pdf1./factor;
probability = normalized_pdf1;

end