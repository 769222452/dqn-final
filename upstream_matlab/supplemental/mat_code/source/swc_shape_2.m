function [x_end,y_end,z_end,y_fct,z_fct,v_end]=swc_shape_2(M,x_v_r,y_v_r,z_v_r,d_fct,x_design,dz,ddz)
    x_end(1:M)=x_design;
    d = d_fct;
    a1=0;
    syms u v real
    eq1 = (a1*y_v_r^2+d)-z_v_r;
    eq2 = x_design-x_v_r;
    [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0 1.5; 0 Inf]);          
    u_s = eval(u_s); v_s = eval(v_s);
    % v_end = linspace(0.0001,v_s,M);
    v_end = linspace(v_s,pi/2,M);
     % v_end = [linspace(v_s,0.4*pi,M/2),linspace(0.41*pi,0.5*pi,M/2)];
    syms u v real
    eq = x_design-x_v_r;
    [u_s] = vpasolve(eq,u);
    for i = 1:M
        v = v_end(i);
        u1 = eval(u_s);
        u_end(i) = u1(find(u1>=0&u1<=1.5));
    end
    for i = 1:M
        u = u_end(i); v=v_end(i);
        y_end(i) = eval(y_v_r);
        z_end(i) = eval(z_v_r);
    end
    % y_fct = linspace(y_end(1),y_end(M),M);
    y_fct = [linspace(y_end(1),0.09,M*0.5),linspace(0.08,y_end(M),M*0.5)];
    z_fct(1:M) = a1*y_fct.^2+d;
    % z_fct(1:M)=d;
    for i = 1:M
        u = u_end(i); v = v_end(i);
        afa(i) = eval(y_end(i)-(dz.*(1+dz.^2))./ddz);
        bta(i) = eval(z_end(i)+(1+dz.^2)./ddz);
    end
    figure(2)
    plot(y_end,z_end,'g');
    hold on 
    plot(y_fct,z_fct,'b.');
    hold on 
    plot(afa,bta,'g.');
    axis equal 
    hold on
end