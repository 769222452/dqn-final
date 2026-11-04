function [Volume]=volume(x_y_z_xingxian_upper,x_y_z_xingxian_lower)
        M = length(x_y_z_xingxian_upper(1,1,:));
        N = length(x_y_z_xingxian_upper(:,1,1));
        Volume = 0;
        for i = 1:M-1
            for j = 1:N-1
                point1 = [x_y_z_xingxian_upper(j,1,i),x_y_z_xingxian_upper(j,2,i),x_y_z_xingxian_upper(j,3,i)];
                point2 = [x_y_z_xingxian_upper(j+1,1,i),x_y_z_xingxian_upper(j+1,2,i),x_y_z_xingxian_upper(j+1,3,i)];
                point3 = [x_y_z_xingxian_upper(j+1,1,i+1),x_y_z_xingxian_upper(j+1,2,i+1),x_y_z_xingxian_upper(j+1,3,i+1)];
                point4 = [x_y_z_xingxian_upper(j,1,i+1),x_y_z_xingxian_upper(j,2,i+1),x_y_z_xingxian_upper(j,3,i+1)];
                point5 = [x_y_z_xingxian_lower(j,1,i),x_y_z_xingxian_lower(j,2,i),x_y_z_xingxian_lower(j,3,i)];
                point6 = [x_y_z_xingxian_lower(j+1,1,i),x_y_z_xingxian_lower(j+1,2,i),x_y_z_xingxian_lower(j+1,3,i)];
                point7 = [x_y_z_xingxian_lower(j+1,1,i+1),x_y_z_xingxian_lower(j+1,2,i+1),x_y_z_xingxian_lower(j+1,3,i+1)];
                point8 = [x_y_z_xingxian_lower(j,1,i+1),x_y_z_xingxian_lower(j,2,i+1),x_y_z_xingxian_lower(j,3,i+1)];
                P_x = (point1(1)+point2(1)+point3(1)+point4(1)+point5(1)+point6(1)+point7(1)+point8(1))/8;
                P_y = (point1(2)+point2(2)+point3(2)+point4(2)+point5(2)+point6(2)+point7(2)+point8(2))/8;
                P_z = (point1(3)+point2(3)+point3(3)+point4(3)+point5(3)+point6(3)+point7(3)+point8(3))/8;
                point_P = [P_x,P_y,P_z];
                V1 = volume_wumianti(point_P,point1,point2,point3,point4);
                V2 = volume_wumianti(point_P,point2,point6,point7,point3);
                V3 = volume_wumianti(point_P,point3,point7,point8,point4);
                V4 = volume_wumianti(point_P,point4,point8,point5,point1);
                V5 = volume_wumianti(point_P,point1,point5,point6,point2);
                V6 = volume_wumianti(point_P,point5,point6,point7,point8);
                V = V1+V2+V3+V4+V5+V6;
                Volume = Volume+V;
            end
        end
end