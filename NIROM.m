classdef NIROM
    properties (SetAccess = public)
        
        X; % input parameter
        U; % Snapshots
        PODmodel; % POD class
        PredModel; % M x 1 (class) KrigingModel
        DvBoundary;
        
    end
    
    methods
        
        function obj=init_(obj,X,U,DvBoundary)
            obj.X=X;
            obj.U=U;
            obj.DvBoundary=DvBoundary;
        end
        
        function obj=train(obj)
            
            obj.PODmodel=POD;
            obj.PODmodel=obj.PODmodel.init_(obj.U, {"energy", 99.9});
            alpha_train=obj.PODmodel.transform(obj.U);
            obj.PredModel=cell(obj.PODmodel.N_mode,1);
            for i=1:obj.PODmodel.N_mode
                obj.PredModel{i}=Kriging;
                obj.PredModel{i}=obj.PredModel{i}.init_(obj.X, alpha_train(:,i), obj.DvBoundary);
                obj.PredModel{i}=obj.PredModel{i}.train();
            end
        end
        
        function y=pred(obj,point)
            N_p=length(point(:,1));
            alpha_pred=zeros(N_p, obj.PODmodel.N_mode);
            for i=1:N_p
                for j=1:obj.PODmodel.N_mode
                    alpha_pred(i,j)=obj.PredModel{j,1}.pred(point(i,:));
                end
            end
            y=obj.PODmodel.inv_transform(alpha_pred);
        end
        
        
        function y=pred_loo(obj,point,point_loo)
            N_p=length(point(:,1));
            alpha_pred=zeros(N_p, obj.PODmodel.N_mode);
            for i=1:N_p
                for j=1:obj.PODmodel.N_mode
                    alpha_pred(i,j)=obj.PredModel{j,1}.pred_loo(point(i,:), point_loo);
                end
            end
            y=obj.PODmodel.inv_transform(alpha_pred);
        end
        
        function y=MSE(obj,point)
            temp=0;
            E=obj.PODmodel.Energy(1:obj.PODmodel.N_mode);
            E=E/sum(E);
            
            for i=1:obj.PODmodel.N_mode
                temp=temp+E(i)*obj.PredModel{i,1}.MSE(point);
            end
            y=temp;
        end
        
        function y=MSE_1st(obj,point) % first dominant Kriging model, variance
            temp=0;
            E=obj.PODmodel.Energy(1:1);
            E=E/sum(E);
            
            for i=1:1%obj.PODmodel.N_mode
                temp=temp+E(i)*obj.PredModel{i,1}.MSE(point);
            end
            y=temp;
        end
        
        function y=MSE_norm(obj,point) 
            temp=0;
            E=obj.PODmodel.Energy(1:obj.PODmodel.N_mode);
            E=E/sum(E);
            
            for i=1:obj.PODmodel.N_mode
                temp=temp+E(i)*obj.PredModel{i,1}.MSE_norm(point);
            end
            y=temp;
        end
        
        function y=MSE_norm_1st(obj,point) % first dominant Kriging model, normlized variance
            temp=0;
            E=obj.PODmodel.Energy(1:1);
            E=E/sum(E);
            
            for i=1:1%obj.PODmodel.N_mode
                temp=temp+E(i)*obj.PredModel{i,1}.MSE_norm(point);
            end
            y=temp;
        end
        
    end
    
    
end