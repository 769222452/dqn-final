function G = v7_prepare_geometry(lo, up)
% Panel/strip topology cache for a closed half-body given as native 50x50x3 lower/upper grids (native frame; body = diag(1,-1,-1)).
% Cache is keyed by the SHA-256 of the coordinates; a changed geometry always rebuilds.
persistent cache; if isempty(cache), cache = containers.Map(); end
key = v7_hash(lo, up);
if isKey(cache,key), G = cache(key); return; end
[ni,nj,~] = size(lo); nr = ni-1; nc = nj-1;
Ns = cell(1,4); As = cell(1,4); Rs = cell(1,4); k = 0;
for s = 1:2
    if s==1, P = cat(3, lo(:,:,1), -lo(:,:,2), -lo(:,:,3)); sgn = -1; else, P = cat(3, up(:,:,1), -up(:,:,2), -up(:,:,3)); sgn = 1; end
    p1 = P(1:end-1,1:end-1,:); p2 = P(2:end,1:end-1,:); p3 = P(2:end,2:end,:); p4 = P(1:end-1,2:end,:);
    v = sgn*cross(p3-p1, p2-p4, 3); mag = sqrt(sum(v.^2,3));
    if any(mag(:)<=1e-14), error('v7:geometry','degenerate quad'); end
    n = v./mag; c = (p1+p2+p3+p4)/4;
    for mir = [1 -1]
        k = k+1; Ns{k} = n.*reshape([1 mir 1],1,1,3); As{k} = mag/2; Rs{k} = c.*reshape([1 mir 1],1,1,3);
    end
end
G.nr = nr; G.nc = nc; G.n = cat(4,Ns{:}); G.A = cat(3,As{:}); G.R = cat(4,Rs{:});   % n:[nr nc 3 4]; A:[nr nc 4]
G.surface = [1 1 2 2]; G.mirror = [1 -1 1 -1];                                    % block -> 1 lower / 2 upper
% strip arc length (distance from the strip leading edge to the panel centre along the grid lines)
G.s = zeros(nr,nc,4); G.s1 = G.s; G.s2 = G.s;
for b = 1:4
    if G.surface(b)==1, P = cat(3, lo(:,:,1), lo(:,:,2), lo(:,:,3)); else, P = cat(3, up(:,:,1), up(:,:,2), up(:,:,3)); end
    Mid = (P(1:end-1,:,:)+P(2:end,:,:))/2;                      % mid polyline of each strip, [nr nj 3]
    seg = sqrt(sum(diff(Mid,1,2).^2,3)); se = [zeros(nr,1) cumsum(seg,2)];
    G.s(:,:,b) = (se(:,1:end-1)+se(:,2:end))/2; G.s1(:,:,b) = se(:,1:end-1); G.s2(:,:,b) = se(:,2:end);
    G.dx_min(b) = min(diff(Mid(:,:,1),1,2),[],'all');
end
G.foldback = any(G.dx_min < -1e-12);                             % strip folds back in x -> flagged
% exact area vector of the open base (full body): strip between lower and upper trailing curves (native frame)
B = [squeeze(lo(:,end,:)); flipud(squeeze(up(:,end,:)))];      %#ok<NASGU>
a_vec = zeros(1,3);
for i = 1:ni-1
    q1 = lo(i,end,:); q2 = lo(i+1,end,:); q3 = up(i+1,end,:); q4 = up(i,end,:);
    q1 = squeeze(q1)'; q2 = squeeze(q2)'; q3 = squeeze(q3)'; q4 = squeeze(q4)';
    a_vec = a_vec + 0.5*(cross(q3-q1, q2-q4));                    % quad area vector
end
G.base_area_half = norm(a_vec); G.base_area = 2*norm(a_vec);
G.base_n = [1 0 0];                                              % outward (rearward) normal in body frame, x downstream
G.wetted_area = 2*(sum(G.A(:,:,1),'all')+sum(G.A(:,:,3),'all'))/1;   % blocks 1 and 3 are the +y halves of lower/upper? (see below)
G.wetted_area = sum(G.A,'all');
G.hash = key; G.Aref_planform = NaN;
cache(key) = G;
end
function k = v7_hash(lo, up)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(lo(:)),'uint8')); md.update(typecast(double(up(:)),'uint8'));
b = typecast(md.digest(),'uint8'); k = lower(reshape(dec2hex(b)',1,[]));
end
