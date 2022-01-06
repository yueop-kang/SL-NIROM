function y=voronoi_index(point,inputdata,DvBoundary)

N_DV=length(inputdata(1,:));

for i=1:N_DV
    point(:,i)=(point(:,i)-DvBoundary(i,2))/(DvBoundary(i,1)-DvBoundary(i,2));
end

for i=1:N_DV
    inputdata(:,i)=(inputdata(:,i)-DvBoundary(i,2))/(DvBoundary(i,1)-DvBoundary(i,2));
end


min_value=10^10;
min_index=0;
N=length(inputdata(:,1));
dist=zeros(1,N);
% for i=1:length(inputdata(:,1))
% dist(i)=distance_krig(point,inputdata(i,:),shape);
% end

for i=1:N
    temp=0;
    for j=1:length(inputdata(1,:))
        temp=temp+(point(j)-inputdata(i,j))^2;
    end
    dist(i)=temp;
end

for i=1:N
    if dist(i)<=min_value
       min_value=dist(i);
       min_index=i;
    end
        
end

if min_index==0
    ['Warning: minimal value of distance_kriging is over 10^10']
end

y=min_index;

end