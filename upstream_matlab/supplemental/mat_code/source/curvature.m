function [r,vector]=curvature(x,y)
%     n = 8;
%     Y=polyfit(x,y,n);
%     syms t f(t)
%     f(t)=poly2sym(Y,t);    
%     dy=diff(f(t),t);
%     ddy=diff(f(t),t,2);
%     t=x;
%     x1 = [x(2) x(1) x x(end) x(end-1)];
%     y1 = [-y(2) -y(1) y -y(end) -y(end-1)];
%     dy1 = gradient(y1,x1);
%     ddy1 = gradient(dy1,x1);
%     dy = dy1(3:end-2);
%     ddy = ddy1(3:end-2);
%     bta1 = x-(dy.*(1+dy.^2))./ddy;
%     bta2 = y+(1+dy.^2)./ddy;
    bta1 = (x(1)+x(end))/2; bta2 = (y(1)+y(end))/2;
    vector(:,1)=(x-bta1)';
    vector(:,2)=(y-bta2)';
    r=sqrt((y-bta2).^2+(x-bta1).^2);
end