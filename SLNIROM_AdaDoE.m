function xnew_save=SLNIROM_AdaDoE(F,LOOCV_error_initial, adjF)
% F is SLNIROM class
optimizer_options_AdaDoE=[length(F.X(1,:)),15,100,100,0.8,0.1,2];
N_infill=length(adjF);
%% Adaptive DoE
% Initialization
LOOCV_add=LOOCV_error_initial;
N_C=F.num_cluster;
MLR_model=F.MLRmodel;
X=F.X;
N_DV=length(X(1,:));
DvBoundary=F.DvBoundary;
ovl_threshold=F.ovl_threshold;

KM_add=cell(N_C,1);
for i=1:N_C
    KM_add{i,1}=F.NIROMmodel{i,1}.PredModel;
end

% NIROM_add=cell(N_C,1);
% for i=1:N_C
%     NIROM_add{i,1}=NIROM;
%     NIROM_add{i,1}=NIROM_add{i,1}.init_(F.NIROMmodel{i,1}.POD_basis,KM_add{i,1},F.NIROMmodel{i,1}.Mean);
% end

F_add=F;
F_add=F_add.init_LOOCV(LOOCV_add,adjF(1));

xnew_save=[];

% iteration start
for iter=1:N_infill
    
    fprintf(['Adaptive DoE (',num2str(iter),'/',num2str(N_infill),')\n'])
    xnew=Optimizer_GA(optimizer_options_AdaDoE,"linear",@F_add.LOOCV_MSE_opt);
    xnew=xnew';
    
    for i=1:N_DV % scale inv
        xnew(:,i)=xnew(:,i)*(DvBoundary(i,1)-DvBoundary(i,2))+DvBoundary(i,2);
    end
    
    LOOCV_add=[LOOCV_add; 0];
    W_add=MLR_model.pred(xnew);
    idx_assign_add=fcm_assign(W_add,0.01); % this threshold has to be much lower than fcm_threshold. If not, added samples can be clustered too much.
    
    xnew_save=[xnew_save; xnew];
    
    % update model for sequential AdaDoE
    for i=1:N_C
        if idx_assign_add{i}==1
            for k=1:length(F.NIROMmodel{i,1}.PredModel)
                clearvars input_new output_new
                input_new=[F_add.NIROMmodel{i,1}.PredModel{k,1}.input; xnew];
                output_new=[F_add.NIROMmodel{i,1}.PredModel{k,1}.output;F_add.NIROMmodel{i,1}.PredModel{k,1}.pred(xnew)];
                F_add.NIROMmodel{i,1}.PredModel{k,1}= F_add.NIROMmodel{i,1}.PredModel{k,1}.init_model(input_new,output_new);
                
            end
        end
    end
    F_add.X=[F_add.X; xnew]; % for voronoi cell
    
    if iter<N_infill
        F_add=F_add.init_LOOCV(LOOCV_add,adjF(iter+1));
    end
end
F_add=F_add.init_LOOCV(LOOCV_add,adjF(iter));
end

