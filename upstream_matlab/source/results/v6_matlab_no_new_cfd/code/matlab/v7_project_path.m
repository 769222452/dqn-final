function [resolved, CORE_ROOT] = v7_project_path(entry)
% Resolve project-relative data entries from this file, never from pwd.
% Absolute entries are retained; provenance/metadata relocation is separate.
here = fileparts(mfilename('fullpath'));
study = fileparts(fileparts(here));
CORE_ROOT = fileparts(fileparts(study));
entry = char(entry);
if isempty(entry), error('v7:path','Empty data path'); end
entry = strrep(strrep(entry,'\',filesep),'/',filesep);
if ~isempty(regexp(entry,'^([A-Za-z]:[\\/]|[\\/])','once'))
    resolved = entry;
else
    resolved = fullfile(CORE_ROOT,entry);
end
end
