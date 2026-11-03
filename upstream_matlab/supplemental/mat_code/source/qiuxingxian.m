
function [xingxian,x1,y1,p1,r1,v1,a1,v1_x,v1_y,det_x] = qiuxingxian(x_shock,r_shock,h_center_body,p0,a0,v0,r0,shock_angle)

global g
global xishu
g=1.4;n=2;xishu=1;
m0 = v0./sqrt(g*p0./r0);
bojiao=nan(1,n);x=bojiao;y=bojiao;p=bojiao;r=bojiao;v=bojiao;a=bojiao; 

for i=1:n
    x(1,i)=x_shock(i); y(1,i)=r_shock(i)+h_center_body;
end
for i=1:n
    bojiao(1,i)=shock_angle(1,i);%%%激波角
    p(1,i)=(2*g/(g+1)*(m0(i)*sin(bojiao(1,i)))^2-(g-1)/(g+1))*p0(i);%%%%激波上点压力
    r(1,i)=(g+1)*(m0(i)*sin(bojiao(1,i)))^2/(2+(g-1)*(m0(i)*sin(bojiao(1,i)))^2)*r0(i);%%%%激波上点密度
    pianzhuanjiao=bojiao(1,i)-atan(2*tan(bojiao(1,i))*(1/m0(i)^2/(sin(bojiao(1,i)))^2+(g-1)/2)/(g+1));%%%%激波上点流动偏转角
    a(1,i)=pianzhuanjiao+a0(i);%%%%激波上点流动偏转角,实际上为速度V与X轴的夹角
    v(1,i)=sin(bojiao(1,i))/sin(bojiao(1,i)-pianzhuanjiao)*(2/(g+1)/(m0(i)*sin(bojiao(1,i)))^2+(g-1)/(g+1))*v0(i);%%%%激波上点速度
end
x1=nan(n,n);y1=x1;p1=x1;r1=x1;v1=x1;a1=x1;m1=x1;v1_x=x1;v1_y=x1;det_x=x1;
for i=1:n
    x1(i,i)=x(1,i);
    y1(i,i)=y(1,i);
    p1(i,i)=p(1,i);
    r1(i,i)=r(1,i);
    v1(i,i)=v(1,i);
    a1(i,i)=a(1,i);
end
for i=2:n
    for j=i-1:-1:1     
        [x1(i,j),y1(i,j),p1(i,j),r1(i,j),v1(i,j),a1(i,j),det_x(i,j)]=reverse_shock(x1(i-1,j),y1(i-1,j),p1(i-1,j),r1(i-1,j),v1(i-1,j),a1(i-1,j),x1(i,j+1),y1(i,j+1),p1(i,j+1),r1(i,j+1),v1(i,j+1),a1(i,j+1));       
       
    end
end
for i=1:n
    for j=1:i
        m1(i,j)=v1(i,j)/sqrt(g*p1(i,j)/r1(i,j)); 
    end
end
for i=1:n
    for j=1:i
        v1_x(i,j)=v1(i,j)*cos(a1(i,j));v1_y(i,j)=v1(i,j)*sin(a1(i,j));
    end
end

for i=1:n
    xingxian(i,1)=x1(i,1);
    xingxian(i,2)=y1(i,1)-h_center_body;
    
end
end













