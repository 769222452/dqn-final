function [x_end,y_end,z_end,y_fct,z_fct,v_end]=swc_shape(M,x_v_r,y_v_r,z_v_r,d_fct,x_design,dz,ddz)
    x_end(1:M)=x_design;
    d = d_fct;
    syms u v real
    eq1 = (0.15*y_v_r^2+0.4)-z_v_r;
    eq2 = x_design-x_v_r;
    [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0 1.5; 0 1]);          
    u_s = eval(u_s); v_s = eval(v_s);
    v_end = linspace(min(v_s),(max(v_s)+min(v_s))/2,M);
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
    y_fct = linspace(y_end(1),y_end(M),M);
    z_fct(1:M) = 0.15*y_fct.^2+0.4;
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