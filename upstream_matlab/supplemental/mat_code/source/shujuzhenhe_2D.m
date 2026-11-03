function [ data,np,nodes,Nelayer ] = shujuzhenhe_2D( x3,y3,z3,p3,vx3,vy3,vz3,p1,han,lie)
%% 关联矩阵
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%自定义X轴、Y轴上的节点数
 Nnx=lie;
 Nny=han;
 Nnz=2; %三维时，需要真实的单元数
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nex=Nnx-1;      %每个水平层X轴上有多少单元数
Ney=Nny-1;      %每个水平层Y轴上有多少单元数
Nez=Nnz-1;      %每个垂直层Z轴上有多少单元数
Nelayer=Nex*Ney;  %每层单元数
Nnlayer=Nnx*Nny;  %每层结点数
Nele=Nelayer*Nez; %总的单元数
Nn=Nnlayer*Nnz;   %总的节点数

elements = [1 : Nele]';
np = zeros(Nele,4);%三维时，需要的是np = zeros(Nele,8);

remainderLe = mod(elements,Nelayer);        %按照层数，存储每层单元的索引号
remainderLe(find(remainderLe==0)) = Nelayer;%去掉零，整理好每层单元的索引编号
remainderXe = mod(remainderLe,Nex);       
remainderXe(find(remainderXe==0)) = Nex;    %按照层数，为每层的x、y编号

np(:,1) = (ceil(elements/Nelayer)-1) * Nnlayer + (ceil(remainderLe/Nex)-1) * Nnx + remainderXe;
np(:,2) = np(:,1) + 1;
np(:,3) = np(:,2) + Nnx;
np(:,4) = np(:,1) + Nnx;
% np(:,5:8) = np(:,1:4) + Nnlayer;          %三维时需要，各个单元的编号集合
%% 为导出tecplot数据做准备
% data=zeros(han*lie,4);
data=zeros(han*lie,7);
lm=1;
for i=1:han
    for j=1:lie
    data(lm,1)=x3(i,j);
    data(lm,2)=y3(i,j);
    data(lm,3)=z3(i,j);
    data(lm,4)=p3(i,j)/p1;
    data(lm,5)=vx3(i,j);
    data(lm,6)=vy3(i,j);
    data(lm,7)=vz3(i,j);
    lm=lm+1;
    end
end  
nodes=han*lie;
end
