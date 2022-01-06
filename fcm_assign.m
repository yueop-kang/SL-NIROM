function [idx_assign, w_filter] = fcm_assign(w, ovl_threshold)
% Input: Cluster: K x N Matrix, K: Cluster index, N: Sample index
% Output: Assigned index (type: cell)

N_C=length(w(:,1));
N_S=length(w(1,:));
idx_assign=cell(N_C,1);
fcm_temp=zeros(1,N_C);
w_filter=zeros(N_C,N_S);

for j=1:N_S
    for i=1:N_C
        if w(i,j)>ovl_threshold || N_C==1
            fcm_temp(i)=fcm_temp(i)+1;
            idx_assign{i}(fcm_temp(i),1)=j;
            w_filter(i,j)=w(i,j);
        else
            w_filter(i,j)=0;
        end
    end
    
    if N_C>1
        [maxval, maxindex]=max(w(:,j));
        if maxval<=ovl_threshold            
            ['Cluster weight of ',num2str(j),'th Snapshot is less than fcm_threshold !!']
            fcm_temp(maxindex)=fcm_temp(maxindex)+1;            
            idx_assign{maxindex}(fcm_temp(maxindex),1)=j;
            w_filter(i,j)=w(i,j);
        end
    end
    w_filter(:,j)=w_filter(:,j)/sum(w_filter(:,j));
end

end