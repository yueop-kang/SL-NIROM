classdef SLNIROM % Soft local non-intrusive reduced-order modeling
    properties (SetAccess = public)
        X; % Input parmaeter
        X_sub;
        U; % Snapshot
        U_sub;
        DvBoundary;
        DvBoundary_sub;
        num_cluster;
        ovl_threshold;
        NIROMmodel;
        W; % membership degree/class label
        MLRmodel;
        ensemble_mode;
        LOOCV;
        adj;
    end
    
    methods
        
        function obj=init_(obj,X,U,DvBoundary,num_cluster,ovl_threshold)
%             obj.NIROMmodel=NIROMmodel;
%             obj.MLRmodel=MLRmodel;
            obj.X=X;
            obj.U=U;
            obj.DvBoundary=DvBoundary;
            obj.num_cluster=num_cluster;
            obj.ovl_threshold=ovl_threshold;
            obj.ensemble_mode="soft_decay";
        end
        
        function obj=init_LOOCV(obj,LOOCV,adj)
            obj.LOOCV=LOOCV;
            obj.adj=adj;
        end
        
        function obj=fcm_split(obj)
            fprintf(['Soft clustering using FCM... \n'])
            obj.X_sub=cell(obj.num_cluster,1);
            obj.U_sub=cell(obj.num_cluster,1);
            
            if obj.num_cluster == 1
                w_min=ones(1,length(obj.U(:,1)));
            else
                fcm_options=[1.8 1000 10^(-5) false];
                POD_fcm=POD;
                POD_fcm=POD_fcm.init_(obj.U, {"mode", 2});
                alpha_fcm=POD_fcm.transform(obj.U);
                J_obj_min=10^10;
                for i=1:500
                    [mu,w_temp,J_fcm] = fcm(alpha_fcm,obj.num_cluster,fcm_options);
                    if J_fcm(length(J_fcm))<J_obj_min
                        J_obj_min=J_fcm(length(J_fcm));
                        w_min=w_temp;
                        mu_min=mu;
                    end
                end
            end
            
            fprintf(['Dataset assignment to each cluster... \n'])
            [idx_assign, obj.W] = fcm_assign(w_min, obj.ovl_threshold);
            
            for i=1:obj.num_cluster
                obj.X_sub{i}=obj.X(idx_assign{i},:);
                obj.U_sub{i}=obj.U(idx_assign{i},:);
            end
        end
        
        function obj=train_NIROM(obj)
            
            obj.NIROMmodel=cell(obj.num_cluster,1);
            obj.DvBoundary_sub=cell(obj.num_cluster,1);
            for i=1:obj.num_cluster
                for j=1:length(obj.X(1,:))
                    if max(obj.X_sub{i}(:,j)) == min(obj.X_sub{i}(:,j))
                        obj.DvBoundary_sub{i}(j,:)=[max(obj.X_sub{i}(:,j)) 0];
                    else
                        obj.DvBoundary_sub{i}(j,:)=[max(obj.X_sub{i}(:,j)) min(obj.X_sub{i}(:,j))];
                    end
                end 
            end
            
            for i=1:obj.num_cluster
                fprintf(['Training individual NIROMs (',num2str(i),'/',num2str(obj.num_cluster),')\n'])
                obj.NIROMmodel{i}=NIROM;
                obj.NIROMmodel{i}=obj.NIROMmodel{i}.init_(obj.X_sub{i},obj.U_sub{i}, obj.DvBoundary_sub{i});
                obj.NIROMmodel{i}=obj.NIROMmodel{i}.train();
            end
        end
        
        function obj=train_MLR(obj)
           
            p_order_table=[2];
            reg_step=[-4:0.05:-2];
            reg_table=10.^reg_step;
            obj.MLRmodel=MLR;
            obj.MLRmodel=obj.MLRmodel.init_(obj.X,obj.W,obj.DvBoundary);
            obj.MLRmodel=obj.MLRmodel.tuning_hyperparameter(p_order_table, reg_table);
            
        end
        
        function y=pred(obj,point)
            [K, D]=size(obj.NIROMmodel);
            L=length(obj.NIROMmodel{1,1}.pred(point));
            y=zeros(1,L);
            ensembleW = obj.MLRmodel.pred(point);
            ensembleW_filter = filter_weight(ensembleW, obj.ovl_threshold, obj.ensemble_mode);
            
            for i=1:K
                y(1,:)=y(1,:)+ensembleW_filter(i)*obj.NIROMmodel{i,1}.pred(point);
            end
        end
        
        function y=pred_loo(obj,point,point_loo)
            [K, D]=size(obj.NIROMmodel);
            L=length(obj.NIROMmodel{1,1}.pred(point));
            y=zeros(1,L);
            ensembleW = obj.MLRmodel.pred_loo(point,point_loo);
            ensembleW_filter = filter_weight(ensembleW, obj.ovl_threshold, obj.ensemble_mode);

            for i=1:K
                y(1,:)=y(1,:)+ensembleW_filter(i)*obj.NIROMmodel{i,1}.pred_loo(point,point_loo);
            end
        end
        
        
        function y=MSE(obj,point)
            [K, ~]=size(obj.NIROMmodel);
            temp=0;
            ensembleW = obj.MLRmodel.pred(point);
            ensembleW_filter = filter_weight(ensembleW, obj.ovl_threshold, obj.ensemble_mode);
            for i=1:K
%               temp=temp+ensembleW_filter(i)^2*obj.NIROMmodel{i,1}.MSE_1st(point);
                temp=temp+ensembleW_filter(i)^2*obj.NIROMmodel{i,1}.MSE_norm_1st(point);
            end
            y=temp;
        end
        
        
        function y=LOOCV_MSE_opt(obj,point)
            
            point=point'; % for Optimizer_GA
            % scale inv
            for i=1:length(point(1,:))
                point(:,i)=point(:,i)*(obj.DvBoundary(i,1)-obj.DvBoundary(i,2))+obj.DvBoundary(i,2);
            end
            
            [K, ~]=size(obj.NIROMmodel);
            temp=0;
            ensembleW = obj.MLRmodel.pred(point);
            ensembleW_filter = filter_weight(ensembleW, obj.ovl_threshold, obj.ensemble_mode);
            for i=1:K
%               temp=temp+ensembleW_filter(i)^2*obj.NIROMmodel{i,1}.MSE_1st(point);
                temp=temp+ensembleW_filter(i)^2*obj.NIROMmodel{i,1}.MSE_norm_1st(point);
            end
            y=obj.LOOCV(voronoi_index(point,obj.X,obj.DvBoundary))^obj.adj*temp;
            y=-y; % for optimizer
        end
    end
end