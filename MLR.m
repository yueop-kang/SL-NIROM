classdef MLR % Multinomial logistic regression
    properties (SetAccess = private)
        
        
        X; % features: N x D, N: The number of data, D: The number of feature
        W; % Soft label: K x N , K: The number of class
        N_C;
        N_S;
        DvBoundary;
        lambda;
        p_order;
        phi;
        Theta;
        llh;
        
    end
    
    methods
        
        function obj=init_(obj,X,W,DvBoundary)
            obj.X=X;
            obj.W=W;
            obj.DvBoundary=DvBoundary;
            obj.N_C=length(obj.W(:,1));
            obj.N_S=length(obj.W(1,:));
        end
        
        function obj=train(obj, p_order, lambda)
            obj.p_order=p_order;
            obj.lambda=lambda;
            phi_temp=poly_expand(obj.X,obj.DvBoundary,p_order);
            obj.phi=phi_temp;

            if obj.N_C>1
                [obj.Theta,obj.llh]=logitMn(phi_temp', obj.W, lambda);
            end
        end
        
        function obj=tuning_hyperparameter(obj, p_order_table, reg_table)
            % k-fold cross validation to tune the hyper-parameter
            kfold=4;
            
            
            X_kfold=cell(kfold,1);
            W_kfold=cell(kfold,1);
            N_kfold=fix(obj.N_S/kfold)+1;
            
            idx=randperm(obj.N_S);
            idx_fold=cell(kfold,1);
            kfold_pred=cell(kfold,1);
            
            for k=1:kfold
                if k~=kfold
                    idx_fold{k}=idx((k-1)*N_kfold+1:(k)*N_kfold);
                else
                    idx_fold{k}=idx((k-1)*N_kfold+1:obj.N_S);
                end
                X_kfold{k}=obj.X(idx_fold{k},:);
                W_kfold{k}=obj.W(:,idx_fold{k});
            end
            
            %             p_order_table=[2];
            %             reg_step=[-4:0.05:-2];
            %             reg_table=10.^reg_step;
            MLR_LOOCV=cell(length(p_order_table),length(reg_table));
            RSQ=zeros(length(p_order_table),length(reg_table));
            
            for i=1:length(p_order_table)
                for j=1:length(reg_table)
                    MLR_LOOCV{i,j}=zeros(obj.N_C,obj.N_S);
                end
            end
            
            MLRmodel_kfold=MLR;
            p=0;
            min_rsq=10^10;
            error_kfold_table=zeros(length(p_order_table),length(reg_table));
            error_kfold=zeros(1,kfold);
            
            for i=1:length(p_order_table)
                for j=1:length(reg_table)
                    p=p+1;
                    fprintf(['MLR Hyperparameter Tuning: (',num2str(p),'/',num2str(length(reg_table)*length(p_order_table)),')\n'])
                    for k=1:kfold
                        
                        X_dummy=[];
                        W_dummy=[];
                        
                        for kk=1:kfold
                            if kk~=k
                                W_dummy=[W_dummy W_kfold{kk}];
                                X_dummy=[X_dummy; X_kfold{kk}];
                            end
                        end
                        
                        MLRmodel_kfold=MLRmodel_kfold.init_(X_dummy,W_dummy,obj.DvBoundary);
                        MLRmodel_kfold=MLRmodel_kfold.train(p_order_table(i),reg_table(j));
                        
                        for m=1:length(X_kfold{k}(:,1))
                            kfold_pred{k}(:,m)=MLRmodel_kfold.pred(X_kfold{k}(m,:));
                        end
                        
                        % test
                        error_kfold(k)=sum(sum((W_kfold{k}-kfold_pred{k}).^2)/obj.N_C)/N_kfold;
                        entropy_kflod(k)=-sum(sum(W_kfold{k}.*log(kfold_pred{k})));
                    end
                    error_kfold_table(i,j)=sum(error_kfold)/kfold;
                    entropy_kfold_table(i,j)=sum(entropy_kflod)/kfold;
                    error_kfold_table_filter(i,j)=sum(error_kfold)/kfold;
                    
                    
                    if entropy_kfold_table(i,j)<min_rsq
                        min_rsq=entropy_kfold_table(i,j);
                        min_index=[i,j];
                    end
                end
            end
            
            obj.p_order=p_order_table(min_index(1));
            obj.lambda=reg_table(min_index(2));
            obj=obj.train(obj.p_order,obj.lambda);
            
        end
        
        
        function y=pred(obj,point)
            if obj.N_C==1
                y=1;
            else
                phi_pred=poly_expand(point,obj.DvBoundary,obj.p_order);
                [~,y] = logitMnPred(obj.Theta, phi_pred');
            end
        end
        
        
        function y=pred_loo(obj,point,point_loo)
            if obj.N_C==1
                y=1;
            else
                N=length(obj.X(:,1));
     
                
                loo_index=0;
                for i=1:obj.N_S
                    if obj.X(i,:)==point_loo
                        loo_index=i;
                    end
                end
                
                X_loo=obj.X;
                phi_loo=obj.phi;
                W_loo=obj.W;
                
                if loo_index>0
                    N=N-1;
                    X_loo(loo_index,:)=[];
                    phi_loo(loo_index,:)=[];
                    W_loo(:,loo_index)=[];
                end
                
                phi_loo=poly_expand(X_loo,obj.DvBoundary,obj.p_order);
                Theta_loo=logitMn(phi_loo', W_loo, obj.lambda);
                
                phi_pred=poly_expand(point,obj.DvBoundary,obj.p_order);
                [~,y] = logitMnPred(Theta_loo, phi_pred');
            end
        end
    end
    
    
end