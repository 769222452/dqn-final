function [Vq]=interpt_3D(x_inter,y_inter,z_inter,v_inter,Xq,Yq,Zq)
    index = 8;
    X=[x_inter(:,:)];
    Y=[y_inter(:,:)];
    Z=[z_inter(:,:)];
    V=[v_inter(:,:)];
%     plot3(X,Y,Z,'.','markersize',12)
    D = sqrt((X-Xq).^2+(Y-Yq).^2+(Z-Zq).^2);
    t = sort(D(:));
    [m,n]=find(D<=t(index),index);
    xishu = 0;
    for i = 1:index
        xishu = xishu+1/(D(m(i),n(i)))^2;
    end
%     Vq1 = griddata(X,Y,Z,V,Xq,Yq,Zq);
    Vq2 = 0;
    for i = 1:index
        Vq2 = Vq2+(1/(D(m(i),n(i)))^2)*V(m(i),n(i))/xishu;
    end
    Vq = Vq2;
end