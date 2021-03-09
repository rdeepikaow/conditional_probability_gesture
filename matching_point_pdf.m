function [probability] = matching_point_pdf(matching_point_TRRS)

filename = 'results/matching_point_pdf_parameters_sf_3.mat';
data = load(filename);
s = data.s;
pd1 = makedist('Normal','mu',1,'sigma',s);
pdf1 = pdf(pd1,matching_point_TRRS);

pdf1max = pdf(pd1,1);
probability = pdf1./pdf1max;

end