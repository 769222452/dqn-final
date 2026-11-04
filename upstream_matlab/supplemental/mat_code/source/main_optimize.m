clc
clear
format short
if isempty(gcp('nocreate'))
    parpool('local',40);
end
[pareto_front, design_vars] = surrogate_assisted_ga_wave_rider_6();