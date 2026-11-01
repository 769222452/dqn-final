function [rat]=rotate_matrix(a,b) % the rotate matrix for 'rat*a=b'
    b=b/norm(b); a=a/norm(a); % �������������λ��
    rotate_angle=acos(dot(a,b)); % ���������֮��ļн�
    rotate_axis=cross(a,b); % �����ת�ᣬ������������ƽ��ķ�����
    rotate_axis=rotate_axis/norm(rotate_axis); % ��ת��������λ��
    matrix=zeros(3,3);
    matrix(1,2)=-rotate_axis(3);
    matrix(1,3)=rotate_axis(2);
    matrix(2,1)=rotate_axis(3);
    matrix(2,3)=-rotate_axis(1);
    matrix(3,1)=-rotate_axis(2);
    matrix(3,2)=rotate_axis(1);
    rat=eye(3)+(1-cos(rotate_angle))*matrix*matrix+sin(rotate_angle)*matrix; % Ӧ���޵����ʽ�����ת����
    if isnan(rat)==1
        rat=eye(3);
    end
end