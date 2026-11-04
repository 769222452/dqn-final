function [Fx_func, Fy_func, Fz_func, xu_func, yu_func, zu_func, xv_func, yv_func, zv_func, ...
          n0x_func, n0y_func, n0z_func, nux_func, nuy_func, nuz_func, nvx_func, nvy_func, nvz_func, ...
          dz_bf_func, ddz_bf_func, dz_func, ddz_func, ...
          x_v_func, y_v_func, z_v_func, x_v_r_func, y_v_r_func, z_v_r_func] = ...
         create_shock_functions(angle_shock_a1,angle_shock_a2,angle_shock_b1,angle_shock_b2,a1,b1,x_design,rat)
% 创建所有必要的数值计算函数
% 输入参数:
%   Ri: 半径
%   beta: 角度(度)
%   c1, c0: 系数
%   rat: 3x3旋转矩阵

%% 符号定义阶段 (只执行一次)
syms u v real

% 定义基本方程
% x_v = u;
% y_v = (Ri + tand(beta)*u) .* ((c1*u - c0).*(u - 1) + u) .* cos(v);
% z_v = (Ri + tand(beta)*u) .* sin(v);
x_v = u*x_design;
y_v = (u*x_design*(u*tand(angle_shock_a2)-u*tand(angle_shock_a1)+tand(angle_shock_a1))+a1)*cos(v);
z_v = (u*x_design*(u*tand(angle_shock_b2)-u*tand(angle_shock_b1)+tand(angle_shock_b1))+b1)*sin(v);

% 添加坐标函数 ========================================================
% 旋转前坐标函数
x_v_func_base = matlabFunction(x_v, 'Vars', {u, v});
y_v_func_base = matlabFunction(y_v, 'Vars', {u, v});
z_v_func_base = matlabFunction(z_v, 'Vars', {u, v});

% 旋转后坐标
x_y_z_v = rat * [x_v; y_v; z_v];
x_v_r = x_y_z_v(1);
y_v_r = x_y_z_v(2);
z_v_r = x_y_z_v(3);

% 旋转后坐标函数
x_v_r_func_base = matlabFunction(x_v_r, 'Vars', {u, v});
y_v_r_func_base = matlabFunction(y_v_r, 'Vars', {u, v});
z_v_r_func_base = matlabFunction(z_v_r, 'Vars', {u, v});
% ====================================================================

% 旋转前曲率方程
xu_bf = diff(x_v, u);
yu_bf = diff(y_v, u);
zu_bf = diff(z_v, u);
xv_bf = diff(x_v, v);
yv_bf = diff(y_v, v);
zv_bf = diff(z_v, v);

% 分支处理 - 转换为函数
% is_xv_bf_zero = isAlways(xv_bf == 0,'Unknown','falseWithWarning');

% if is_xv_bf_zero
if isAlways(xv_bf == 0)
    dz_bf = zv_bf / yv_bf;
    ddz_bf = diff(dz_bf, v) / yv_bf;
else
    dz_bf = (zu_bf + (-xu_bf/xv_bf)*zv_bf) / (yu_bf + (-xu_bf/xv_bf)*yv_bf);
    ddz_bf = (diff(dz_bf, u) + (-xu_bf/xv_bf)*diff(dz_bf, v)) / ...
             (yu_bf + (-xu_bf/xv_bf)*yv_bf);
end

% 旋转后导数
xu = diff(x_v_r, u);
yu = diff(y_v_r, u);
zu = diff(z_v_r, u);
xv = diff(x_v_r, v);
yv = diff(y_v_r, v);
zv = diff(z_v_r, v);

F_vec = cross([xu; yu; zu], [xv; yv; zv]);
Fx = F_vec(1);
Fy = F_vec(2);
Fz = F_vec(3);

% 分支处理 - 转换为函数
% is_xv_zero = isAlways(xv == 0,'Unknown','falseWithWarning');

if isAlways(xv == 0)
% if is_xv_zero
    dz = zv / yv;
    ddz = diff(dz, v) / yv;
else
    dz = (zu + (-xu/xv)*zv) / (yu + (-xu/xv)*yv);
    ddz = (diff(dz, u) + (-xu/xv)*diff(dz, v)) / ...
          (yu + (-xu/xv)*yv);
end

% 法向量计算
F_norm = sqrt(Fx^2 + Fy^2 + Fz^2);
n0x = Fx / F_norm;
n0y = Fy / F_norm;
n0z = Fz / F_norm;

% 法向量导数
nux = diff(n0x, u); nuy = diff(n0y, u); nuz = diff(n0z, u);
nvx = diff(n0x, v); nvy = diff(n0y, v); nvz = diff(n0z, v);

%% 创建数值函数 (高效计算)
% 坐标函数 ===========================================================
x_v_func = x_v_func_base;
y_v_func = y_v_func_base;
z_v_func = z_v_func_base;
x_v_r_func = x_v_r_func_base;
y_v_r_func = y_v_r_func_base;
z_v_r_func = z_v_r_func_base;
% ===================================================================

% F 分量
Fx_func = matlabFunction(Fx, 'Vars', {u, v});
Fy_func = matlabFunction(Fy, 'Vars', {u, v});
Fz_func = matlabFunction(Fz, 'Vars', {u, v});

% 旋转后导数
xu_func = matlabFunction(xu, 'Vars', {u, v});
yu_func = matlabFunction(yu, 'Vars', {u, v});
zu_func = matlabFunction(zu, 'Vars', {u, v});
xv_func = matlabFunction(xv, 'Vars', {u, v});
yv_func = matlabFunction(yv, 'Vars', {u, v});
zv_func = matlabFunction(zv, 'Vars', {u, v});

% 法向量
n0x_func = matlabFunction(n0x, 'Vars', {u, v});
n0y_func = matlabFunction(n0y, 'Vars', {u, v});
n0z_func = matlabFunction(n0z, 'Vars', {u, v});

% 法向量导数
nux_func = matlabFunction(nux, 'Vars', {u, v});
nuy_func = matlabFunction(nuy, 'Vars', {u, v});
nuz_func = matlabFunction(nuz, 'Vars', {u, v});
nvx_func = matlabFunction(nvx, 'Vars', {u, v});
nvy_func = matlabFunction(nvy, 'Vars', {u, v});
nvz_func = matlabFunction(nvz, 'Vars', {u, v});

% 旋转前曲率
dz_bf_func = matlabFunction(dz_bf, 'Vars', {u, v});
ddz_bf_func = matlabFunction(ddz_bf, 'Vars', {u, v});

% 旋转后曲率
dz_func = matlabFunction(dz, 'Vars', {u, v});
ddz_func = matlabFunction(ddz, 'Vars', {u, v});
end
