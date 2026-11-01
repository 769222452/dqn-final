function [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(Ma, angle_shock_a1, angle_shock_a2, angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA, a_xishu_1)
    % 通过旋转激波来处理，贝塞尔曲面
    % 乘波体分为两个部分，前者保证激波，后者保证体积+重构各流面内的网格(指定流面出口参数)
    % % 关闭符号计算的警告
    % warning('on', 'symbolic:sym:sym:DeprecateExpressions');
    % warning('on', 'symbolic:generate:UnableToProve');
    % % 关闭插值函数中的重复点警告
    % warning('on', 'MATLAB:scatteredInterpolant:DupPtsAvValuesWarnId');
    % warning('on','all');
    tic
    M = 50; N_l = 40; nn=M;
    % 开启并行池
    % if isempty(gcp('nocreate'))
    %     parpool('local');
    % end
    % 设置数值优化选项（在所有工作进程上相同）
    options = optimoptions('fsolve', ...
        'Display', 'off', ...
        'Algorithm', 'levenberg-marquardt', ...
        'FunctionTolerance', 1e-10, ...
        'StepTolerance', 1e-10, ...
        'MaxIterations',200);    
    %%%%%%%%%%%%%%%%%%%%%%%%来流条件%%%%%%%%%%%%%%%%%%%%%%%%%%
    global p0 r0 m0 v0 a0 a_xishu % 定义出口边界的偏转角发展系数
    p0=79.779;r0=0.001027;m0=Ma;v0=m0*sqrt(1.4*p0/r0); a0=0;
    Vector_V=[cosd(AOA) 0 sind(AOA)]'; % AOA为攻角，AOS为侧滑角，Vector_V表示来流方向
    rat=rotate_matrix(Vector_V,[1 0 0]'); % rat为旋转矩阵，rat*a=b, inv（rat）*b=a；
    a_xishu=a_xishu_1;
    x_design=5; 
    bezier_params = struct();
    bezier_params.control_points = [
        0.0, 1.0;    % P0: Y=0, Z_norm=1.0（起点高）
        0.5, 1.0;    % P1: Y=0.4, Z_norm=1.0（第一个中间控制点）
        0.8, 0.2;    % P2: Y=1.2, Z_norm=0.3（第二个中间控制点）
        1.0, 0.0     % P3: Y=2.0, Z_norm=0.0（终点低）
    ];
    det_H_sym = 0.42; % 对称面处，出口截面上下表面的高度差
    up_cst_options = struct();
    up_cst_options.nx = nn; 
    up_cst_options.L = x_design;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%激波形状%%%%%%%%%%%%%%%%%%%%%%%%%
    % u = linspace(0,1,100); v = linspace(0,pi/2,100);
    % for i = 1:100
    %     for j = 1:100
    %         x(j,i)=u(i)*x_design;  
    %         y(j,i)=(u(i)*x_design*(u(i)*tand(angle_shock_a2)-u(i)*tand(angle_shock_a1)+tand(angle_shock_a1))+c1)*cos(v(j));
    %         z(j,i)=(u(i)*x_design*(u(i)*tand(angle_shock_b2)-u(i)*tand(angle_shock_b1)+tand(angle_shock_b1))+c2)*sin(v(j));
    %         x_y_z_xuanzhuan=rat*[x(j,i),y(j,i),z(j,i)]';
    %         x_xuanzhuan(j,i)=x_y_z_xuanzhuan(1); y_xuanzhuan(j,i)=x_y_z_xuanzhuan(2); z_xuanzhuan(j,i)=x_y_z_xuanzhuan(3);
    %     end
    %     btoa(i)=y(1,i)/z(100,i);
    % end
    % figure(1)
    % mesh(x,y,z,'edgecolor','k','facecolor','none') % 旋转前激波形状
    % axis equal
    % hold on
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 创建数值计算函数 (只需执行一次)
    [Fx_func, Fy_func, Fz_func, xu_func, yu_func, zu_func, xv_func, yv_func, zv_func, ...
     n0x_func, n0y_func, n0z_func, nux_func, nuy_func, nuz_func, nvx_func, nvy_func, nvz_func, ...
     dz_bf_func, ddz_bf_func, dz_func, ddz_func, ...
     x_v_func, y_v_func, z_v_func, x_v_r_func, y_v_r_func, z_v_r_func] = ...
        create_shock_functions(angle_shock_a1,angle_shock_a2,angle_shock_b1,angle_shock_b2,c1,c2,x_design,rat);
    %%%%%%%%%%%%%%%%%%%%%%%%%前缘形状%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 定义前缘形状时使用的是旋转前，定义好后旋转再计算
    % 保证旋转前的前缘形状不变，因此定义时只需要指定fct的u,v就行
    % 定义前缘形状时使用的是旋转前，定义好后旋转再计算
    %% 使用二阶贝塞尔曲线定义前缘型线的水平投影PLF
    N = [1,-2,1; -2,2,0; 1,0,0]; % 二次贝塞尔矩阵
    p00 = [0,0]; p01 = [0,d1]; p02 = [1,0.2];
    P_X = [p00(1),p01(1),p02(1)];
    P_Y = [p00(2),p01(2),p02(2)];
    transition_point = 0.09; % 过渡点
    % 第一部分：从起点到过渡点
    idx1 = 1:floor(M*0.5);
    t(idx1) = linspace(0, transition_point, length(idx1));
    det_y = abs(t(1)-t(2));
    % 第二部分：从过渡点到终点
    idx2 = (floor(M*0.5)+1):M;
    t(idx2) = linspace(transition_point+det_y, 1, length(idx2));
    t = fliplr(t);
    % 预分配数组
    u_fct = zeros(1, M);
    v_fct = zeros(1, M);
    x_fct_1 = zeros(1, M);
    y_fct_1 = zeros(1,M);
    z_fct_1 = zeros(1,M);
    deta_x = zeros(1, M);
    x_fct = zeros(1, M);
    y_fct = zeros(1, M);
    z_fct = zeros(1, M);    
    % 创建搜索网格
    u_guesses = linspace(0, 1.1, 30);
    v_guesses = linspace(0, pi/2, 30);
    [U_grid, V_grid] = meshgrid(u_guesses, v_guesses);    
    % 并行计算所有前缘点
    for i = 1:M
        try
            % 获取当前目标值
            x_fct_1(i)=x_design*[t(i)^2,t(i),1]*N*P_X';
            y_fct_1(i)=x_design*[t(i)^2,t(i),1]*N*P_Y';
            target_x = x_fct_1(i);
            % target_z = z_fct_1(i);
            target_y = y_fct_1(i);            
            % 在网格上计算残差（避免符号计算）
            residuals = zeros(size(U_grid));
            for idx = 1:numel(U_grid)
                u_val = U_grid(idx);
                v_val = V_grid(idx);                
                % 使用数值函数计算坐标
                calc_x = x_v_func(u_val, v_val);
                % calc_z = z_v_func(u_val, v_val);
                calc_y = y_v_func(u_val, v_val);                
                % 计算残差
                residuals(idx) = abs(calc_x - target_x) + abs(calc_y - target_y);
            end
            
            % 找到最小残差的位置
            [~, min_idx] = min(residuals(:));
            u_init = U_grid(min_idx);
            v_init = V_grid(min_idx);            
            % 使用数值优化精化解
            % options = optimoptions('fsolve', 'Display', 'off', ...
            %                       'Algorithm', 'levenberg-marquardt', ...
            %                       'FunctionTolerance', 1e-8, ...
            %                       'StepTolerance', 1e-8);            
            [solution,residual,flag] = fsolve(@(x) equations(x, target_y, target_x, y_v_func, x_v_func), ...
                              [u_init, v_init], options);            
            u_s = solution(1);
            v_s = solution(2);            
            % 确保结果在合理范围内
            if flag<=0 || norm(residual,inf)>1e-7 || u_s<0 || u_s>1.1 || v_s<0 || v_s>pi/2
                error('V5:GeometryNonconverged','Leading edge root invalid');
            end            
            % 存储结果
            u_fct(i) = u_s;
            v_fct(i) = v_s;            
            % 计算其他量
            % x_fct_1(i) = x_v_func(u_s, v_s);
            z_fct_1(i) = z_v_func(u_s, v_s);
            deta_x(i) = ((x_design + 0.1 - x_fct_1(i)) / (N_l - 1)) * cosd(AOA);
            x_fct(i) = x_v_r_func(u_s, v_s);
            y_fct(i) = y_v_r_func(u_s, v_s);
            z_fct(i) = z_v_r_func(u_s, v_s);            
        catch ME
            rethrow(ME);
        end
    end    
    % 辅助函数：定义方程组
    function F = equations(uv, target_y, target_x, y_v_func, x_v_func)
        u_val = uv(1);
        v_val = uv(2);
        F = zeros(2, 1);
        F(1) = x_v_func(u_val, v_val) - target_x;
        F(2) = y_v_func(u_val, v_val) - target_y;
    end

    % figure(1)
    % plot3(x_fct_1,y_fct_1,z_fct_1,'g.');
    % hold on  
    % 预分配大型数组变量
    x_y_z_shock = zeros(N_l+1, 3, M);
    x_r_shock = zeros(N_l+1, 3, M);
    U_index = zeros(M, N_l+1);
    V_index = zeros(M, N_l+1);
    shock_angle = zeros(M, N_l+1);
    beta_index = zeros(N_l+1, N_l+1, M);
    h_center_body = zeros(M, N_l+1);
    x_l = zeros(N_l+1, N_l+1, M); y_l = x_l; p_l = x_l; r_l = x_l; v_l = x_l; a_l = x_l;
    v_x = x_l; v_y = x_l; v_z = x_l;
    x_inter = x_l; y_inter = x_l; z_inter = x_l;
    h_r_inter = x_l; h_inter = x_l;
    h1 = x_l; h2 = x_l; x_index = x_l;
    index = zeros(1, M);
    x_l_2 = zeros(N_l+2,N_l+1,M); y_l_2 = x_l_2; p_l_2 = x_l_2; r_l_2 = x_l_2; v_l_2 = x_l_2; a_l_2 = x_l_2;
    x_inter_2 = x_l_2; y_inter_2 = x_l_2; z_inter_2 = x_l_2;
    v_x_2 = x_l_2; v_y_2 = x_l_2; v_z_2 = x_l_2; beta_index_2 = x_l_2;
    x_l_3 = zeros(2*N_l+1, N_l+1, M); y_l_3 = x_l_3; p_l_3 = x_l_3; r_l_3 = x_l_3; v_l_3 = x_l_3; a_l_3 = x_l_3;
    x_inter_3 = x_l_3; y_inter_3 = x_l_3; z_inter_3 = x_l_3;
    v_x_3 = x_l_3; v_y_3 = x_l_3; v_z_3 = x_l_3;
    wav_x = zeros(M,nn); wav_y = wav_x; wav_z = wav_x; 
    p_wav = wav_x; vx_wav = wav_x; vy_wav = wav_x; vz_wav = wav_x;
    x_y_z_wav = zeros(nn,3,M);
    wav_up_x = wav_x; wav_up_y = wav_x; wav_up_z = wav_x; p_wav_up = wav_x; 
    x_y_z_up = x_y_z_wav; x_y_z_wav_r = x_y_z_wav; x_y_z_up_r = x_y_z_wav;
    wav_x_new=zeros(M,nn); wav_y_new=zeros(M,nn); wav_z_new=zeros(M,nn);
    leading_edge_points = struct();
    leading_edge_points.X_LE = zeros(M,1); leading_edge_points.Y_LE = zeros(M,1); leading_edge_points.Z_LE = zeros(M,1);
    for i=1:M
    % for i=1:M

        try
            % 初始化每轮迭代的临时变量
            local_U_index = NaN(1, N_l+1);
            local_V_index = NaN(1, N_l+1);
            local_x_y_z_shock = zeros(N_l+1, 3);
            local_x_r_shock = zeros(N_l+1, 3);
            local_shock_angle = zeros(1, N_l+1);
            local_beta_index = zeros(N_l+1, N_l+1);
            local_h_center_body = zeros(1, N_l+1);
            local_x_l = zeros(N_l+1, N_l+1); local_y_l = local_x_l; local_p_l = local_x_l; local_r_l = local_x_l; 
            local_v_l = local_x_l; local_a_l = local_x_l;
            local_v_x = local_x_l; local_v_y = local_x_l; local_v_z = local_x_l;
            local_x_inter = local_x_l; local_y_inter = local_x_l; local_z_inter = local_x_l;
            local_h_r_inter = local_x_l; local_h_inter = local_x_l; 
            local_h1 = local_x_l; local_h2 = local_x_l; local_x_index = local_x_l;
            local_x_l_2 = zeros(N_l+2,N_l+1); local_y_l_2 = local_x_l_2; 
            local_p_l_2 = local_x_l_2; local_r_l_2 = local_x_l_2;
            local_v_l_2 = local_x_l_2; local_a_l_2 = local_x_l_2;
            local_v_x_2 = local_x_l_2; local_v_y_2 = local_x_l_2; local_v_z_2 = local_x_l_2;
            local_x_inter_2 = local_x_l_2; local_y_inter_2 = local_x_l_2; local_z_inter_2 = local_x_l_2;  
            local_beta_index_2 = local_x_l_2; local_h1_2 = local_x_l_2; local_h2_2 = local_x_l_2;
            local_x_index_2 = local_x_l_2;
            local_x_y_z_xingxian = zeros(2*N_l+1, 3); local_x_y_z_xingxian_r = local_x_y_z_xingxian;
            local_p_wav_1 = local_x_y_z_xingxian;
            local_vx_wav_1 = local_x_y_z_xingxian; local_vy_wav_1 = local_x_y_z_xingxian; local_vz_wav_1 = local_x_y_z_xingxian;
            local_wav_x=zeros(1,nn); local_wav_y=local_wav_x; local_wav_z=local_wav_x;
            local_p_wav = local_wav_x; local_vx_wav = local_wav_x; local_vy_wav = local_wav_x; 
            local_vz_wav = local_wav_x; local_x_y_z_wav = zeros(nn,3);
            local_wav_up_x = local_wav_x; local_wav_up_y = local_wav_x; local_wav_up_z = local_wav_x; 
            local_p_wav_up = local_wav_x; local_x_y_z_up = local_x_y_z_wav;
            local_index = 0; local_x_fct = x_fct_1(i); local_y_fct = y_fct_1(i); local_z_fct = z_fct_1(i);
            % 复制全局变量到局部（避免广播问题）
            local_p0 = p0; local_r0 = r0; local_m0 = m0; local_v0 = v0; local_a0 = a0;
            local_a_xishu = a_xishu; 
            % 计算初始点
            x_s = x_fct(i); y_s = y_fct(i); z_s = z_fct(i);
            u_val = u_fct(i); v_val = v_fct(i);            
            local_x_y_z_shock(1,:) = [x_s, y_s, z_s];
            local_U_index(1,1) = u_val; local_V_index(1,1) = v_val;
            Fx_val = Fx_func(u_val, v_val); Fy_val = Fy_func(u_val,v_val); Fz_val = Fz_func(u_val,v_val);
            F_norm = norm([Fx_val, Fy_val, Fz_val]); ratio = min(abs(Fx_val)/F_norm, 1);
            local_shock_angle(1,1) = pi/2-acos(ratio);
            local_beta_index(1,1) = included_angle([Fy_val,Fz_val]);
            local_x_r_shock(1,1) = local_x_y_z_shock(1,1);
            local_x_r_shock(1,2) = sqrt((local_x_y_z_shock(1,2)-local_x_y_z_shock(1,2)).^2+(local_x_y_z_shock(1,3)-local_x_y_z_shock(1,3)).^2);
            % 计算第一点后激波参数
            F_val = [Fx_val, Fy_val, Fz_val];
            vector_C=cross(F_val,Vector_V); vector_C=vector_C/norm(vector_C);
            vector_D=cross(F_val,vector_C); vector_D=vector_D/norm(vector_D);
            ru=[xu_func(u_val,v_val),yu_func(u_val,v_val),zu_func(u_val,v_val)]; 
            rv=[xv_func(u_val,v_val),yv_func(u_val,v_val),zv_func(u_val,v_val)];
            nu_s=[nux_func(u_val,v_val),nuy_func(u_val,v_val),nuz_func(u_val,v_val)]; 
            nv_s=[nvx_func(u_val,v_val),nvy_func(u_val,v_val),nvz_func(u_val,v_val)];
            Sb=calculate_H(ru,rv,nu_s,nv_s,vector_D); 
            R=abs(-cos(local_shock_angle(1,1))/Sb);
            local_h_center_body(1,1)=R; local_x_r_shock(1,3)=R;
            h_real = local_h_center_body(1,1)-local_x_r_shock(1,2);
            local_h_r_inter(1,1)=h_real;
            local_h_inter(1,1)=local_h_center_body(1,1);
            [x1,y1,p1,r1,v1,a1,v1_x,v1_r] = qiujie_shock( ...
                local_x_r_shock(1,1),local_x_r_shock(1,2),h_real, ...
                local_p0,local_a0,local_v0,local_r0,local_shock_angle(1,1));
            local_x_l(1,1)=x1; local_y_l(1,1)=y1; local_p_l(1,1)=p1; 
            local_r_l(1,1)=r1; local_v_l(1,1)=v1; local_a_l(1,1)=a1;
            local_v_x(1,1)=v1_x; 
            local_v_y(1,1)=v1_r*cos(local_beta_index(1,1));
            local_v_z(1,1)=v1_r*sin(local_beta_index(1,1));
            local_x_inter(1,1)=local_x_y_z_shock(1,1);
            local_y_inter(1,1)=local_x_y_z_shock(1,2);
            local_z_inter(1,1)=local_x_y_z_shock(1,3);
            local_x_l(1,2:N_l+1)=local_x_l(1,1); local_y_l(1,2:N_l+1)=local_y_l(1,1); 
            local_p_l(1,2:N_l+1)=local_p_l(1,1); local_r_l(1,2:N_l+1)=local_r_l(1,1);
            local_v_l(1,2:N_l+1)=local_v_l(1,1); local_a_l(1,2:N_l+1)=local_a_l(1,1);
            local_x_inter(1,2:N_l+1)=local_x_inter(1,1); local_y_inter(1,2:N_l+1)=local_y_inter(1,1); local_z_inter(1,2:N_l+1)=local_z_inter(1,1);
            local_v_x(1,2:N_l+1)=local_v_x(1,1); local_v_y(1,2:N_l+1)=local_v_y(1,1); local_v_z(1,2:N_l+1)=local_v_z(1,1);
            local_beta_index(1,2:N_l+1)=local_beta_index(1,1);
            % 开始计算内点
            for j=2:N_l+1
                u_val = local_U_index(1,j-1); v_val = local_V_index(1,j-1);
                Fx_val = Fx_func(u_val,v_val); Fy_val = Fy_func(u_val,v_val); Fz_val = Fz_func(u_val,v_val);
                vector_D = 0.02*[Fz_val^2+Fy_val^2,-Fx_val*Fy_val,-Fx_val*Fz_val];
                delta_x = deta_x(i);
                delta_y = delta_x*vector_D(2)/vector_D(1);
                local_x_y_z_shock(j,1) = local_x_y_z_shock(j-1,1)+delta_x;
                local_x_y_z_shock(j,2) = local_x_y_z_shock(j-1,2)+delta_y;
                target_x = local_x_y_z_shock(j,1); target_y = local_x_y_z_shock(j,2);
                % 使用前一点作为初始值
                u_init = local_U_index(1,j-1);
                v_init = local_V_index(1,j-1);     
                eq_func = @(uv) [
                    x_v_r_func(uv(1), uv(2)) - target_x;
                    y_v_r_func(uv(1), uv(2)) - target_y;
                ];
                [solution,residual,exitflag] = fsolve(eq_func,[u_init,v_init],options);
                u_s=solution(1); v_s=solution(2);
                if exitflag<=0 || norm(residual,inf)>1e-7 || u_s<0 || u_s>2 || v_s<0 || v_s>pi/2
                    error('V5:GeometryNonconverged','Interior shock root invalid');
                end
                local_U_index(1,j) = u_s; local_V_index(1,j) = v_s;
                % 计算z坐标
                u_val = local_U_index(1,j); v_val = local_V_index(1,j);
                try
                    local_x_y_z_shock(j,3) = z_v_r_func(u_val, v_val);
                catch ME
                    rethrow(ME);
                end
                Fx_val = Fx_func(u_val, v_val); Fy_val = Fy_func(u_val, v_val); Fz_val = Fz_func(u_val, v_val);
                F_norm = norm([Fx_val, Fy_val, Fz_val]); ratio = min(abs(Fx_val)/F_norm, 1);
                local_shock_angle(1,j) = pi/2-acos(ratio);
                local_beta_index(j,j) = included_angle([Fy_val,Fz_val]);
                local_x_r_shock(j,1) = local_x_y_z_shock(j,1);
                local_x_r_shock(j,2) = sqrt((local_x_y_z_shock(j,3)-local_x_y_z_shock(j-1,3)).^2+(local_x_y_z_shock(j,2)-local_x_y_z_shock(j-1,2)).^2)+ local_x_r_shock(j-1,2);
                F_val = [Fx_val, Fy_val, Fz_val];
                vector_C=cross(F_val,Vector_V); vector_C=vector_C/norm(vector_C);
                vector_D=cross(F_val,vector_C); vector_D=vector_D/norm(vector_D);
                ru=[xu_func(u_val,v_val),yu_func(u_val,v_val),zu_func(u_val,v_val)]; 
                rv=[xv_func(u_val,v_val),yv_func(u_val,v_val),zv_func(u_val,v_val)];
                nu_s=[nux_func(u_val,v_val),nuy_func(u_val,v_val),nuz_func(u_val,v_val)]; 
                nv_s=[nvx_func(u_val,v_val),nvy_func(u_val,v_val),nvz_func(u_val,v_val)];
                Sb=calculate_H(ru,rv,nu_s,nv_s,vector_D); 
                R=abs(-cos(local_shock_angle(1,1))/Sb); 
                local_h_center_body(1,j)=R; local_x_r_shock(j,3)=R;
                h_real = local_h_center_body(1,j)-local_x_r_shock(j,2);
                local_h_r_inter(j,j)=h_real;
                local_h_inter(j,j)=local_h_center_body(1,j);
                [x1,y1,p1,r1,v1,a1,v1_x,v1_r] = qiujie_shock( ...
                    local_x_r_shock(j,1),local_x_r_shock(j,2),h_real, ...
                    local_p0,local_a0,local_v0,local_r0,local_shock_angle(1,j));
                local_x_l(j,j) = x1; local_y_l(j,j) = y1; local_p_l(j,j) = p1;
                local_v_l(j,j) = v1; local_r_l(j,j) = r1; local_a_l(j,j) = a1;
                local_v_x(j,j) = v1_x;
                local_v_y(j,j) = v1_r*cos(local_beta_index(j,j));
                local_v_z(j,j) = v1_r*sin(local_beta_index(j,j));
                local_x_inter(j,j) = local_x_y_z_shock(j,1);
                local_y_inter(j,j) = local_x_y_z_shock(j,2);
                local_z_inter(j,j) = local_x_y_z_shock(j,3);
                % figure(1)
                % plot3(local_x_y_z_shock(j-1:j,1),local_x_y_z_shock(j-1:j,2),local_x_y_z_shock(j-1:j,3),'r');
                % hidden off
                % hold on
                local_x_l(j,j+1:N_l+1)=local_x_l(j,j); local_y_l(j,j+1:N_l+1)=local_y_l(j,j);
                local_p_l(j,j+1:N_l+1)=local_p_l(j,j); local_r_l(j,j+1:N_l+1)=local_r_l(j,j);
                local_v_l(j,j+1:N_l+1)=local_v_l(j,j); local_a_l(j,j+1:N_l+1)=local_a_l(j,j);
                local_x_inter(j,j+1:N_l+1)=local_x_inter(j,j); local_y_inter(j,j+1:N_l+1)=local_y_inter(j,j); local_z_inter(j,j+1:N_l+1)=local_z_inter(j,j);
                local_v_x(j,j+1:N_l+1)=local_v_x(j,j); local_v_y(j,j+1:N_l+1)=local_v_y(j,j); local_v_z(j,j+1:N_l+1)=local_v_z(j,j);
                local_beta_index(j,j+1:N_l+1)=local_beta_index(j,j);
                for n=j-1:-1:1
                    x1=[];y1=[];p1=[];r1=[];v1=[];a1=[];x2=[];y2=[];p2=[];r2=[];v2=[];a2=[];
                    local_h1(j-1,n) = interp1(local_x_r_shock(1:j,1),local_x_r_shock(1:j,3),local_x_l(j-1,n));
                    local_h2(j-1,n) = interp1(local_x_r_shock(1:j,1),local_x_r_shock(1:j,2),local_x_l(j-1,n));
                    h_real=local_h1(j-1,n)-local_h2(j-1,n);
                    local_h_r_inter(j-1,n)=h_real;
                    local_h_inter(j-1,n)=local_h1(j-1,n);
                    x1 = local_x_l(j-1,n); y1 = local_y_l(j-1,n)+h_real; p1 = local_p_l(j-1,n); 
                    r1 = local_r_l(j-1,n); v1 = local_v_l(j-1,n); a1 = local_a_l(j-1,n);
                    x2 = local_x_l(j,n+1); y2 = local_y_l(j,n+1)+h_real; p2 = local_p_l(j,n+1); 
                    r2 = local_r_l(j,n+1); v2 = local_v_l(j,n+1); a2 = local_a_l(j,n+1);
                    [x3,y3,p3,r3,v3,a3,det3,v3_x,v3_r] = qiujie_inter(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);
                    local_x_l(j,n) = x3; local_y_l(j,n) = y3-h_real; local_p_l(j,n) = p3; 
                    local_a_l(j,n) = a3; local_v_l(j,n) = v3; local_r_l(j,n) = r3;
                    local_x_index(j,n)=det3; local_x_inter(j,n) = local_x_l(j,n);
                    [y4,z4,theta4]=dirction( ...
                        local_x_inter(j-1,n),local_y_inter(j-1,n),local_z_inter(j-1,n), ...
                        local_v_x(j-1,n),local_v_y(j-1,n),local_v_z(j-1,n), ...
                        local_x_inter(j,n+1),local_y_inter(j,n+1),local_z_inter(j,n+1), ...
                        local_v_x(j,n+1),local_v_y(j,n+1),local_v_z(j,n+1), ...
                        local_x_inter(j,n),local_x_index(j,n));
                    local_y_inter(j,n)=y4; local_z_inter(j,n)=z4;
                    local_beta_index(j,n)=theta4;
                    local_v_x(j,n) = v3_x;
                    local_v_y(j,n) = v3_r*cos(local_beta_index(j,n));
                    local_v_z(j,n) = v3_r*sin(local_beta_index(j,n));
                    x3=[];y3=[];p3=[];r3=[];v3=[];v3=[];a3=[];det3=[];v3_x=[];v3_r=[];y4=[];z4=[];theta4=[];
                    det_x=[]; det_y=[]; det_z=[]; Vcetor_V1=[]; Vcetor_V3=[];
                end
                x_distance=local_x_l(j,1); 
               if isnan(x_distance)==1 || x_distance<local_x_inter(j-1,1)
                   error('V5:GeometryDomain','Invalid shock streamline');
               end  
            end
            local_index=j;
            for j=1:2
                local_x_l_2(j,:)=local_x_l(local_index-2+j,:); local_y_l_2(j,:)=local_y_l(local_index-2+j,:);
                local_p_l_2(j,:)=local_p_l(local_index-2+j,:); local_r_l_2(j,:)=local_r_l(local_index-2+j,:);
                local_v_l_2(j,:)=local_v_l(local_index-2+j,:); local_a_l_2(j,:)=local_a_l(local_index-2+j,:);
                local_x_inter_2(j,:)=local_x_inter(local_index-2+j,:); local_y_inter_2(j,:)=local_y_inter(local_index-2+j,:);
                local_z_inter_2(j,:)=local_z_inter(local_index-2+j,:); local_v_x_2(j,:)=local_v_x(local_index-2+j,:);
                local_v_y_2(j,:)=local_v_y(local_index-2+j,:); local_v_z_2(j,:)=local_v_z(local_index-2+j,:);
                local_beta_index_2(j,:)=local_beta_index(local_index-2+j,:);
            end
            for j=1:local_index-1
                for n=local_index-j:-1:1
                    x1=[];y1=[];p1=[];r1=[];v1=[];a1=[];
                    x2=[];y2=[];p2=[];r2=[];v2=[];a2=[];
                    x3=[];y3=[];p3=[];r3=[];v3=[];a3=[];
                    local_h1_2(j+1,n)=interp1(local_x_r_shock(1:local_index,1),local_x_r_shock(1:local_index,3),local_x_l_2(j+1,n));
                    local_h2_2(j+1,n)=interp1(local_x_r_shock(1:local_index,1),local_x_r_shock(1:local_index,2),local_x_l_2(j+1,n));
                    h_real=local_h1_2(j+1,n)-local_h2_2(j+1,n);
                    if n==local_index-j
                        x1 = local_x_l_2(j+1,n); y1 = local_y_l_2(j+1,n)+h_real; p1 = local_p_l_2(j+1,n); r1 = local_r_l_2(j+1,n); v1 = local_v_l_2(j+1,n); a1 = local_a_l_2(j+1,n);
                        x2 = local_x_l_2(j+1,n+1); y2 = local_y_l_2(j+1,n+1)+h_real; p2 = local_p_l_2(j+1,n+1); r2 = local_r_l_2(j+1,n+1); v2 = local_v_l_2(j+1,n+1); a2 = local_a_l_2(j+1,n+1);
                        [x3,y3,p3,r3,v3,a3]=qiujie_end_point_improved(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,local_a_xishu); 
                        local_x_l_2(j+2,n) = x3; local_y_l_2(j+2,n) = y3-h_real; local_p_l_2(j+2,n) = p3; local_a_l_2(j+2,n) = a3; local_v_l_2(j+2,n) = v3; local_r_l_2(j+2,n) = r3;
                        % 坐标变换
                        x1 = local_x_l_2(j,n); y1 = local_y_l_2(j,n);
                        x2 = local_x_l_2(j+1,n+1); y2 = local_y_l_2(j+1,n+1);
                        x3 = local_x_l_2(j+2,n); y3 = local_y_l_2(j+2,n);
                        f12=atan((y2-y1)/(x2-x1)); f03=f12-pi/2;
                        [x0,y0]=intercept(x1,y1,f12,x3,y3,f03);
                        local_x_index_2(j+2,n)=x0-x1;
                        local_x_inter_2(j+2,n)=local_x_l_2(j+2,n);
                        [y4,z4,theta4]=dirction(local_x_inter_2(j,n),local_y_inter_2(j,n),local_z_inter_2(j,n), ...
                            local_v_x_2(j,n),local_v_y_2(j,n),local_v_z_2(j,n), ...
                            local_x_inter_2(j+1,n+1),local_y_inter_2(j+1,n+1),local_z_inter_2(j+1,n+1), ...
                            local_v_x_2(j+1,n+1),local_v_y_2(j+1,n+1),local_v_z_2(j+1,n+1), ...
                            local_x_inter_2(j+2,n),local_x_index_2(j+2,n));
                        local_y_inter_2(j+2,n)=y4; local_z_inter_2(j+2,n)=z4;
                        local_beta_index_2(j+2,n)=theta4;
                        local_v_x_2(j+2,n)=local_v_l_2(j+2,n)*cos(local_a_l_2(j+2,n));
                        local_v_y_2(j+2,n)=local_v_l_2(j+2,n)*sin(local_a_l_2(j+2,n))*cos(local_beta_index_2(j+2,n));
                        local_v_z_2(j+2,n)=local_v_l_2(j+2,n)*sin(local_a_l_2(j+2,n))*sin(local_beta_index_2(j+2,n));
                        x3=[];y3=[];p3=[];r3=[];v3=[];v3=[];a3=[];det3=[];v3_x=[];v3_r=[];y4=[];z4=[];theta4=[];
                        det_x=[]; det_y=[]; det_z=[]; Vcetor_V1=[]; Vcetor_V3=[]; x0=[]; y0=[];                      
                    else
                        x1 = local_x_l_2(j+1,n); y1 = local_y_l_2(j+1,n)+h_real; p1 = local_p_l_2(j+1,n); r1 = local_r_l_2(j+1,n); v1 = local_v_l_2(j+1,n); a1 = local_a_l_2(j+1,n);
                        x2 = local_x_l_2(j+2,n+1); y2 = local_y_l_2(j+2,n+1)+h_real; p2 = local_p_l_2(j+2,n+1); r2 = local_r_l_2(j+2,n+1); v2 = local_v_l_2(j+2,n+1); a2 = local_a_l_2(j+2,n+1);
                        [x3,y3,p3,r3,v3,a3,det3,v3_x,v3_r] = qiujie_inter(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);
                        local_x_l_2(j+2,n) = x3; local_y_l_2(j+2,n) = y3-h_real; local_p_l_2(j+2,n) = p3; local_a_l_2(j+2,n) = a3; local_v_l_2(j+2,n) = v3; local_r_l_2(j+2,n) = r3;
                        local_x_index_2(j+2,n)=det3; local_x_inter_2(j+2,n) = local_x_l_2(j+2,n);
                        [y4,z4,theta4]=dirction(local_x_inter_2(j+1,n),local_y_inter_2(j+1,n),local_z_inter_2(j+1,n), ...
                            local_v_x_2(j+1,n),local_v_y_2(j+1,n),local_v_z_2(j+1,n), ...
                            local_x_inter_2(j+2,n+1),local_y_inter_2(j+2,n+1),local_z_inter_2(j+2,n+1), ...
                            local_v_x_2(j+2,n+1),local_v_y_2(j+2,n+1),local_v_z_2(j+2,n+1), ...
                            local_x_inter_2(j+2,n),local_x_index_2(j+2,n));
                        local_y_inter_2(j+2,n)=y4; local_z_inter_2(j+2,n)=z4;
                        local_beta_index_2(j+2,n)=theta4;
                        local_v_x_2(j+2,n)=v3_x;
                        local_v_y_2(j+2,n)=v3_r*cos(local_beta_index_2(j+2,n));
                        local_v_z_2(j+2,n)=v3_r*sin(local_beta_index_2(j+2,n));
                        x3=[];y3=[];p3=[];r3=[];v3=[];v3=[];a3=[];det3=[];v3_x=[];v3_r=[];y4=[];z4=[];theta4=[];
                        det_x=[]; det_y=[]; det_z=[]; Vcetor_V1=[]; Vcetor_V3=[];
                    end
                end
                local_x_l_2(j+2,local_index-j+1:local_index)=local_x_l_2(j+2,local_index-j); 
                local_y_l_2(j+2,local_index-j+1:local_index)=local_y_l_2(j+2,local_index-j);
                local_p_l_2(j+2,local_index-j+1:local_index)=local_p_l_2(j+2,local_index-j);
                local_r_l_2(j+2,local_index-j+1:local_index)=local_r_l_2(j+2,local_index-j);
                local_v_l_2(j+2,local_index-j+1:local_index)=local_v_l_2(j+2,local_index-j);
                local_a_l_2(j+2,local_index-j+1:local_index)=local_a_l_2(j+2,local_index-j);
                local_x_inter_2(j+2,local_index-j+1:local_index)=local_x_inter_2(j+2,local_index-j);
                local_y_inter_2(j+2,local_index-j+1:local_index)=local_y_inter_2(j+2,local_index-j);
                local_z_inter_2(j+2,local_index-j+1:local_index)=local_z_inter_2(j+2,local_index-j);
                local_v_x_2(j+2,local_index-j+1:local_index)=local_v_x_2(j+2,local_index-j);
                local_v_y_2(j+2,local_index-j+1:local_index)=local_v_y_2(j+2,local_index-j);
                local_v_z_2(j+2,local_index-j+1:local_index)=local_v_z_2(j+2,local_index-j);
                local_beta_index_2(j+2,local_index-j+1:local_index)=local_beta_index_2(j+2,local_index-j);
               x_distance=local_x_l_2(j+2,1);
               if isnan(x_distance)==1 || x_distance<local_x_l_2(j+1,1)
                   error('V5:GeometryDomain','Invalid isentropic streamline');
               end  
            end
            local_x_y_z_xingxian(:,1)=[local_x_inter(:,1); local_x_inter_2(3:end,1)];
            local_x_y_z_xingxian(:,2)=[local_y_inter(:,1); local_y_inter_2(3:end,1)];
            local_x_y_z_xingxian(:,3)=[local_z_inter(:,1); local_z_inter_2(3:end,1)];
            local_p_wav_1=[local_p_l(:,1); local_p_l_2(3:end,1)];
            local_vx_wav_1=[local_v_x(:,1); local_v_x_2(3:end,1)];
            local_vy_wav_1=[local_v_y(:,1); local_v_y_2(3:end,1)];
            local_vz_wav_1=[local_v_z(:,1); local_v_z_2(3:end,1)];
            % local_x_y_z_xingxian_r = inv(rat)*[local_x_y_z_xingxian(:,1),local_x_y_z_xingxian(:,2),local_x_y_z_xingxian(:,3)]';
            local_x_y_z_xingxian_r = rat' * local_x_y_z_xingxian';
            local_wav_x(1,:)=linspace(local_x_y_z_xingxian_r(1,1),x_design,nn);
            local_wav_up_x(1,:)=linspace(local_x_fct,x_design,nn);
            for j=1:nn
                % 插值下表面型线
                local_wav_y(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_x_y_z_xingxian_r(2,:),local_wav_x(1,j));
                local_wav_z(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_x_y_z_xingxian_r(3,:),local_wav_x(1,j));
                local_p_wav(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_p_wav_1(:,1),local_wav_x(1,j));
                local_vx_wav(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_vx_wav_1(:,1),local_wav_x(1,j));
                local_vy_wav(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_vy_wav_1(:,1),local_wav_x(1,j));
                local_vz_wav(1,j)=interp1(local_x_y_z_xingxian_r(1,:),local_vz_wav_1(:,1),local_wav_x(1,j));
                local_x_y_z_wav(j,1)=local_wav_x(1,j); local_x_y_z_wav(j,2)=local_wav_y(1,j); local_x_y_z_wav(j,3)=local_wav_z(1,j);
                % 插值上表面型线
                local_wav_up_y(1,j)=(local_wav_up_x(1,j)-local_wav_up_x(1,1))*Vector_V(2)/Vector_V(1)+local_y_fct;
                local_wav_up_z(1,j)=(local_wav_up_x(1,j)-local_wav_up_x(1,1))*Vector_V(3)/Vector_V(1)+local_z_fct;
                local_p_wav_up(1,j)=local_p0;
                local_x_y_z_up(j,1)=local_wav_up_x(1,j); local_x_y_z_up(j,2)=local_wav_up_y(1,j); local_x_y_z_up(j,3)=local_wav_up_z(1,j);                
            end
            % 存储结果到预分配数组
            U_index(i,:) = local_U_index;
            V_index(i,:) = local_V_index;
            x_y_z_shock(:,:,i) = local_x_y_z_shock;
            beta_index(:,:,i) = local_beta_index;
            x_r_shock(:,:,i) = local_x_r_shock;
            shock_angle(i,:) = local_shock_angle;
            h_center_body(i,:) = local_h_center_body;
            x_l(:,:,i) = local_x_l; y_l(:,:,i) = local_y_l; p_l(:,:,i) = local_p_l;
            r_l(:,:,i) = local_r_l; v_l(:,:,i) = local_v_l; a_l(:,:,i) = local_a_l;
            v_x(:,:,i) = local_v_x; v_y(:,:,i) = local_v_y; v_z(:,:,i) = local_v_z;
            x_inter(:,:,i) = local_x_inter; y_inter(:,:,i) = local_y_inter; z_inter(:,:,i) = local_z_inter;
            h_r_inter(:,:,i) = local_h_r_inter;
            h_inter(:,:,i) = local_h_inter;
            h1(:,:,i) = local_h1;
            h2(:,:,i) = local_h2;
            index(1,i) = local_index;
            x_l_2(:,:,i) = local_x_l_2; y_l_2(:,:,i) = local_y_l_2; p_l_2(:,:,i) = local_p_l_2;
            r_l_2(:,:,i) = local_r_l_2; v_l_2(:,:,i) = local_v_l_2; a_l_2(:,:,i) = local_a_l_2;
            v_x_2(:,:,i) = local_v_x_2; v_y_2(:,:,i) = local_v_y_2; v_z_2(:,:,i) = local_v_z_2;
            x_inter_2(:,:,i) = local_x_inter_2; y_inter_2(:,:,i) = local_y_inter_2; z_inter_2(:,:,i) = local_z_inter_2; 
            x_l_3(:,:,i)=[local_x_l; local_x_l_2(3:end,:)]; y_l_3(:,:,i)=[local_y_l; local_y_l_2(3:end,:)]; p_l_3(:,:,i)=[local_p_l; local_p_l_2(3:end,:)];
            r_l_3(:,:,i)=[local_r_l; local_r_l_2(3:end,:)]; v_l_3(:,:,i)=[local_v_l; local_v_l_2(3:end,:)]; a_l_3(:,:,i)=[local_a_l; local_a_l_2(3:end,:)];
            v_x_3(:,:,i)=[local_v_x; local_v_x_2(3:end,:)]; v_y_3(:,:,i)=[local_v_y; local_v_y_2(3:end,:)]; v_z_3(:,:,i)=[local_v_z; local_v_z_2(3:end,:)];
            x_inter_3(:,:,i)=[local_x_inter; local_x_inter_2(3:end,:)]; 
            y_inter_3(:,:,i)=[local_y_inter; local_y_inter_2(3:end,:)]; 
            z_inter_3(:,:,i)=[local_z_inter; local_z_inter_2(3:end,:)];
            if i==1
                % 保证最外侧只有一个点
                wav_x(i,:)=local_wav_x(1,1); wav_y(i,:)=local_wav_y(1,1); wav_z(i,:)=local_wav_z(1,1);
                p_wav(i,:)=local_p_wav(1,1); vx_wav(i,:)=local_vx_wav(1,1); vy_wav(i,:)=local_vy_wav(1,1); vz_wav(i,:)=local_vz_wav(1,1);
            else
                wav_x(i,:)=local_wav_x; wav_y(i,:)=local_wav_y; wav_z(i,:)=local_wav_z;
                p_wav(i,:)=local_p_wav; vx_wav(i,:)=local_vx_wav; vy_wav(i,:)=local_vy_wav; vz_wav(i,:)=local_vz_wav;
            end
            wav_up_x(i,:)=local_wav_up_x; wav_up_y(i,:)=local_wav_up_y; wav_up_z(i,:)=local_wav_up_z;
            p_wav_up(i,:)=local_p_wav_up;
            x_y_z_wav(:,:,i) = [wav_x(i,:)', wav_y(i,:)', wav_z(i,:)'];
            x_y_z_up(:,:,i) = [wav_up_x(i,:)', wav_up_y(i,:)', wav_up_z(i,:)'];
        catch ME
            error('V5:GeometryFailure','Stream surface %d: %s [%s]',i,ME.message,ME.identifier);
        end
    end    
    fields={wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z};
    for k=1:numel(fields)
        if ~isreal(fields{k}) || any(~isfinite(fields{k}(:)))
            error('V5:GeometryDomain','Nonfinite or complex native surface');
        end
    end
    diagnostics=struct('native_M',M,'native_N',nn,'streamwise_solver_intervals',N_l,'geometry_elapsed_s',toc,'mesh_version','native50x50-stream40-v5','parallel_workers',0,'upper_surface','native generator upper','regridded_for_plot',false);
end
