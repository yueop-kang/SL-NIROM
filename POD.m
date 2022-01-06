classdef POD
    
    properties (SetAccess = private)
        U;
        U_bar;
        U_tilde;
        Basis;
        N_mode;
        Energy;
        
    end
    
    methods
        function obj = init_(obj,U,options)
            obj.U=U;
            obj.U_bar=mean(U,1);
            
            N_S=length(U(:,1));
            N_D=length(U(1,:));
            
            for i=1:N_S
                obj.U_tilde(i,:)=obj.U(i,:)-obj.U_bar;
            end
            
            R=obj.U_tilde*obj.U_tilde';
            [V_tot,D_tot]=eig(R);
            
            lambda=zeros(N_S,1);
            E=zeros(N_S,1);
            
            for i=1:N_S
                lambda(i)=D_tot(i,i);
            end
            
            temp=0;
            for i=1:N_S
                temp=temp+lambda(length(lambda)+1-i)/sum(lambda)*100;
                E(i)=temp;
            end
            
            criteria_option=options{1};
            criteria=options{2};
            
            if criteria_option == "mode"
                obj.N_mode= criteria;
            end
            
            if criteria_option == "energy"
                temp_idx=0;
                for i=1:N_S
                    temp_idx=temp_idx+1;
                    if E(i) >= criteria
                        break
                    end
                end
                obj.N_mode = temp_idx;
            end
            
            
            V=zeros(N_D,obj.N_mode);
            for j=1:obj.N_mode
                V(:,j)=obj.U_tilde'*V_tot(:,length(V_tot(:,1))+1-j);
            end
            
            for j=1:obj.N_mode
                V(:,j)=V(:,j)/norm(V(:,j));
            end
            
            obj.Energy=E;
            obj.Basis=V;
        end
        
        function alpha = transform(obj, data)
            data_tilde=zeros(length(data(:,1)),length(data(1,:)));
            for i=1:length(data(:,1))
                data_tilde(i,:)=data(i,:)-obj.U_bar;
            end
            alpha =  data_tilde*obj.Basis;
        end
        
        function recon = inv_transform(obj, alpha)

            for i=1:length(alpha(:,1))
                recon(i,:) = alpha(i,:)*obj.Basis'+obj.U_bar;
            end
        end
        
    end
end