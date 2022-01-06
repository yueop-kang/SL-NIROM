function DV_expand=poly_expand(DV,DvBoundary,p_order)

N_DV=length(DV(1,:));

for j=1:length(DV(:,1))
    temp(j,1)=1;
end

for i=1:N_DV
    DV(:,i)=(DV(:,i)-DvBoundary(i,2))/(DvBoundary(i,1)-DvBoundary(i,2));
end

for i=1:N_DV
    temp(:,i+1)=DV(:,i);
end

for k=1:p_order-1
    i_temp=0;
    for i=1:length(temp(1,:))
        for j=i:length(temp(1,:))
            i_temp=i_temp+1;
            temp_after(:,i_temp)=temp(:,i).*temp(:,j);
        end
    end
    temp=temp_after;
end

DV_expand=temp;


end