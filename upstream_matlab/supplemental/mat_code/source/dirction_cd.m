function [y3,z3,theta0]=dirction_cd(x1,y1,z1,x2,y2,z2,x0,y0,z0,u0,v0,w0,x3)
    theta0=included_angle([v0,w0]);
    x_j=x3-x0;
    x_i=x2-x1;
    y_i=y2-y1;
    z_i=z2-z1;
    y_j=x_j*(v0*(x_i^2+z_i^2)-u0*x_i*y_i-w0*y_i*z_i)/(u0*(y_i^2+z_i^2)-v0*x_i*y_i-w0*x_i*z_i);
    z_j=-(x_i*x_j+y_i*y_j)/z_i;
    y3=y0+y_j;
    z3=z0+z_j;
    % theta0=atan((z3-z1)/(y3-y1));
end