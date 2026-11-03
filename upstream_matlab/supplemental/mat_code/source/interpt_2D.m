function [Vq]=interpt_2D(X,Y,V,Xq,Yq)
    index = 8;
    D = sqrt((X-Xq).^2+(Y-Yq).^2);
    t = sort(D(:));
    D_min=min(min(D));
    if D_min==0
        [i_min,j_min]=find(D==min(min(D)));
        Vq2=V(i_min,j_min);
    else
        [m,n]=find(D<=t(index),index);
        xishu = 0;
        for i = 1:index
            xishu = xishu+1/(D(m(i),n(i))^1);
        end
        Vq2 = 0;
        for i = 1:index
            Vq2 = Vq2+(1/(D(m(i),n(i))^1))*V(m(i),n(i))/xishu;
        end
    end
    Vq = Vq2;
%     Vq1 = griddata(X,Y,V,Xq,Yq);
end