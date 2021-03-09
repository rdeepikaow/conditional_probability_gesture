function [Pzero,Pacute] = gamma_angle_probability(area_fraction)
% Read the parameters of the gamma distribution from "gamma_parameters.mat"
% file and assign probability of zero angle and acute angle.

filename = 'results/gamma_parameters.mat';
data = load(filename);
gamma1 = data.gamma(1:2);
gamma2 = data.gamma(3:4);

Pzero = gampdf(area_fraction,gamma1(1),gamma1(2));
Pacute = gampdf(area_fraction,gamma2(1),gamma2(2));
normalization_factor = Pzero+Pacute;
Pzero = Pzero/normalization_factor;
Pacute = Pacute/normalization_factor;
end

