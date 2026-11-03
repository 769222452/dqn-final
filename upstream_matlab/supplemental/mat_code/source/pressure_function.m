function [p]=pressure_function(x)
    global a_p
    global b_p
    global c_p
    global d_p
    global x_p    
    p=(a_p*(x-x_p).^3+b_p*(x-x_p).^2+c_p*(x-x_p)+1)*d_p;
    % p=a_p*x^4+b_p*x^3+c_p*x^2+d_p*x+x_p;
end