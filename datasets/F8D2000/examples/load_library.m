function data = load_library(geometry_id)
% Load one F8D2000 geometry and [Mach, alpha] coefficient tables.
if nargin < 1, geometry_id = 'F8D_01351'; end
root = fileparts(fileparts(mfilename('fullpath')));
aero = load(fullfile(root,'aerodynamics','p2_grid.mat'));
ids = string(aero.geometry_ids);
index = find(ids == string(geometry_id));
assert(isscalar(index), 'Unknown or duplicate geometry ID.');
data.geometry_id = char(geometry_id);
data.action_index = index - 1; % CSV/Python index is zero-based.
data.geometry = load(fullfile(root,'geometries',char(geometry_id),'native_geometry.mat'));
data.mach_grid = aero.mach_grid;
data.alpha_deg_grid = aero.alpha_deg_grid;
fields = {'CL','CD_pressure','Cm_pressure','model_valid','usable_p2'};
for k = 1:numel(fields)
    name = fields{k};
    data.(name) = reshape(aero.(name)(index,:,:),numel(aero.mach_grid),numel(aero.alpha_deg_grid));
end
end
