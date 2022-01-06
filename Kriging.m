classdef Kriging
    
    properties (SetAccess = private)
        input;
        input_scaled;
        output;
        DvBoundary;
        theta;
        nugget=10^(-6);
        invR;
        beta;
        M1;
        sigmaSQ;
        
        N_DV;
        N_S;
        optimizer_options; % Dimension, String length, Population, Generation, Selection param., Mutation param., Crossover param.
    end
    
    methods
        function obj = init_(obj,input,output,DvBoundary)
            obj.input=input;
            obj.output=output;
            obj.DvBoundary=DvBoundary;
            obj.N_DV=length(obj.input(1,:));
            obj.N_S=length(obj.input(:,1));
            obj.optimizer_options=[obj.N_DV,15,200,100,0.8,0.1,2]; % Dimension, String length, Population, Generation, Selection param., Mutation param., Crossover param.
            obj.input_scaled=[];
            for i=1:length(obj.input(1,:))
                obj.input_scaled(:,i)=(input(:,i)-DvBoundary(i,2))/(DvBoundary(i,1)-DvBoundary(i,2));
            end
        end
        
        function R = cal_R(obj, input_scaled, theta)
            N=length(input_scaled(:,1));
            nsq=input_scaled.^2*theta;
            R=bsxfun(@minus,nsq,(2* input_scaled)*diag(theta)* input_scaled.');
            R=bsxfun(@plus,nsq.',R);
            R=exp(-R)+obj.nugget*eye(N);
        end
        
        function r = cal_r(obj, point_scaled, input_scaled, theta)
            N=length(input_scaled(:,1));
            N_p=length(point_scaled(:,1));
            r=zeros(N,N_p);
            
            for j=1:N_p
                for i=1:N
                    r(i,j)=exp(-sum(theta'.*(input_scaled(i,:)-point_scaled(j,:)).^2));
                end
            end
        end
            
        function obj = train(obj)
            
            obj.theta=Optimizer_GA(obj.optimizer_options,"log",@obj.Likelihood);
 
            N=length(obj.input(:,1));
            R = obj.cal_R(obj.input_scaled, obj.theta);
            F=ones(N,1);
            Y=obj.output;
            obj.invR=inv(R);
            obj.M1=inv((F'*obj.invR*F));
            obj.beta=obj.M1*F'*obj.invR*Y;
            obj.sigmaSQ=1/N*(Y-F*obj.beta)'*obj.invR*(Y-F*obj.beta);
            
        end
        
        function obj = init_model(obj, input, output)
            % re-initialize input/output without changing correlation
            % parameter (AdaDoE)
            obj.input=input;
            obj.output=output;
            N=length(obj.input(:,1));
            obj.input_scaled=[];
            for i=1:length(obj.input(1,:))
                obj.input_scaled(:,i)=(obj.input(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
            end
            obj.N_S=length(obj.input(:,1));
            R = obj.cal_R(obj.input_scaled, obj.theta);
            F=ones(N,1);
            Y=obj.output;
            obj.invR=inv(R);
            obj.M1=inv((F'*obj.invR*F));
            obj.beta=obj.M1*F'*obj.invR*Y;
            obj.sigmaSQ=1/N*(Y-F*obj.beta)'*obj.invR*(Y-F*obj.beta);
            
        end
        
        function LF = Likelihood(obj,theta)
            N=length(obj.input(:,1));
            % bsxfun formulation
            R = obj.cal_R(obj.input_scaled, theta);   
            F=ones(N,1);
            Y=obj.output;
            invR_temp=inv(R);
            temp_M1=inv((F'*invR_temp*F));
            beta_temp=temp_M1*F'*invR_temp*Y;
            sigmaSQ_temp=1/N*(Y-F*beta_temp)'*invR_temp*(Y-F*beta_temp);
            
            % cost function
            LF=-(N/2)*log(sigmaSQ_temp)-1/2*log(det(R));
            % LF=-(N/2)*log(sigmaSQ)-1/2*log(det(R))-0.001*(sum(theta.^2)+sigmaSQ); %Penalty
            
            if LF>100000000
                LF=-10^4;
            end
            
            LF=-LF;  % for optimizer
        end
        
        function y=pred(obj,point)
            
            point_scaled=point;

            for i=1:obj.N_DV
                point_scaled(:,i)=(point(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
            end
            
            r = obj.cal_r(point_scaled, obj.input_scaled, obj.theta);
            F=ones(obj.N_S,1);    % ordinary kriging
            f(1,1)=1;
            y=f'*obj.beta+r'*obj.invR*(obj.output-F*obj.beta);

        end
        
        
        function y=pred_loo(obj, point, point_loo)
            % Assumption of this function: length(point_loo(:,1)) == 1
            
            N=length(obj.input(:,1));

            loo_index=0;
            for i=1:N
                if obj.input(i,:)==point_loo
                    loo_index=i;
                end
            end
            
            input_loo=obj.input;
            output_loo=obj.output;
            
            if loo_index>0
                N=N-1;
                input_loo(loo_index,:)=[];
                output_loo(loo_index,:)=[];
            end
            
            input_loo_scaled=zeros(N, obj.N_DV);
            point_scaled=point;
            for i=1:obj.N_DV
                point_scaled(:,i)=(point(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
                input_loo_scaled(:,i)=(input_loo(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
            end
            
            F=ones(N,1);    % ordinary kriging
            f(1,1)=1;
            R = obj.cal_R(input_loo_scaled, obj.theta);
            r = obj.cal_r(point_scaled, input_loo_scaled, obj.theta);
            Y=output_loo;
            invR_temp=inv(R);
            temp_M1=inv((F'*invR_temp*F));
            beta_temp=temp_M1*F'*invR_temp*Y;
            sigmaSQ_temp=1/N*(Y-F*beta_temp)'*invR_temp*(Y-F*beta_temp);
            y=f'*beta_temp+r'*invR_temp*(output_loo-F*beta_temp);
            
        end
        
        
        function y=MSE(obj,point)
            point_scaled=point;
            
            for i=1:obj.N_DV
                point_scaled(:,i)=(point(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
            end
            
            F=ones(obj.N_S,1); % ordinary kriging
            f(1,1)=1;
            r = obj.cal_r(point_scaled, obj.input_scaled, obj.theta);
            u=F'*obj.invR*r-f;
            y=obj.sigmaSQ*(1-r'*obj.invR*r+u'*obj.M1*u);
        end
        
        function y=MSE_norm(obj,point)
            point_scaled=point;
            
            for i=1:obj.N_DV
                point_scaled(:,i)=(point(:,i)-obj.DvBoundary(i,2))/(obj.DvBoundary(i,1)-obj.DvBoundary(i,2));
            end
            
            F=ones(obj.N_S,1); % ordinary kriging
            f(1,1)=1;
            r = obj.cal_r(point_scaled, obj.input_scaled, obj.theta);
            u=F'*obj.invR*r-f;
            y=(1-r'*obj.invR*r+u'*obj.M1*u);
        end
    end
end

