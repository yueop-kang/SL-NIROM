clc;clear all;close all;

%% Dataread
dataset_dir='dataset\transonic_training';
dataset_val_dir='dataset\transonic_validation';
N_doe=100;
N_doe_val=400;

DV_org=textread([dataset_dir,'\condition_transonic.txt']);
DV_val=textread([dataset_val_dir,'\condition_transonic.txt']);
x_coord_temp=textread([dataset_dir,'\x_coordinate.dat']);
y_coord_temp=textread([dataset_dir,'\y_coordinate.dat']);

x_coord=x_coord_temp(:,1:end-1);
y_coord=y_coord_temp(:,1:end-1);
x_grid=length(x_coord(:,1));
y_grid=length(y_coord(1,:));

for i=1:N_doe
    filename=[dataset_dir,'\p_',num2str(i),'.dat'];
    temp0=textread(filename);
    temp1=temp0(:,1:end-1);
    Snapshot_reshape(i,:)=temp1(:);
end

for i=1:N_doe_val
    filename=[dataset_val_dir,'\p_',num2str(i),'.dat'];
    temp0=textread(filename);
    temp1=temp0(:,1:end-1);
    Snapshot_val_reshape(i,:)=temp1(:);
end
%% Training of SLNIROM

DvBoundary=[0.9 0.5;3 -2]; % min/max of input parameter
N_C = 4; % number of clusters
ovl_threshold = 0.15; % overlapping parameter
F=SLNIROM; % SLNIROM class
F=F.init_(DV_org,Snapshot_reshape,DvBoundary,N_C,ovl_threshold); % Initialize
F=F.fcm_split(); % Soft clustering and dataset partitioning
F=F.train_NIROM(); % Training individual NIROMs
F=F.train_MLR(); % Training and tuning MLR for combining individual NIROMs

%% Calculate LOOCV

NRMSE=zeros(length(DV_org(:,1)),1);
for i=1:length(DV_org(:,1))
    pred=F.pred_loo(DV_org(i,:),DV_org(i,:));
    NRMSE(i)=sum((Snapshot_reshape(i,:)-pred).^2/(x_grid*y_grid))^0.5/(max(Snapshot_reshape(i,:))-min(Snapshot_reshape(i,:)));
end

NRMSE_val=zeros(length(DV_val(:,1)),1);
for i=1:length(DV_val(:,1))
    pred=F.pred(DV_val(i,:));
    test=Snapshot_val_reshape(i,:);
    NRMSE_val(i)=sum((test-pred).^2/length(test))^0.5/(max(test)-min(test));
end

fprintf(['NRMSE for validation data: ' num2str(mean(NRMSE_val)),'\n'])
%
figure()
set(gcf,'position',[500 200 750 600]);
set(gca,'FontName', 'Times','fontweight','bold','FontSize',18,'LineWidth',1.5);
set(gca,'ColorScale','linear','layer','top')
set(gca,'XTick',[DvBoundary(1,2):(DvBoundary(1,1)-DvBoundary(1,2))/4:DvBoundary(1,1)])
set(gca,'YTick',[DvBoundary(2,2):(DvBoundary(2,1)-DvBoundary(2,2))/5:DvBoundary(2,1)])
set(gca,'XMinorTick','on','YMinorTick','on','TickLength',[0.02 0.02])

hold on
box on
M_size=15;
M_thickness=1.2;
val_xgrid=reshape(DV_val(:,1),[20, 20]);
val_ygrid=reshape(DV_val(:,2),[20, 20]);
NRMSE_val_grid=reshape(NRMSE_val,[20, 20]);
surf(val_xgrid,val_ygrid,NRMSE_val_grid)
view(2)
shading interp
colorbar
caxis([0 0.05])
axis([DvBoundary(1,2) DvBoundary(1,1) DvBoundary(2,2) DvBoundary(2,1)])
xlabel('DV1')
ylabel('DV2')
title('Contour of Validation Error')
%% Model Weight Vis.
N_show=200;
xshow=[DvBoundary(1,2):(DvBoundary(1,1)-DvBoundary(1,2))/N_show:DvBoundary(1,1)];
yshow=[DvBoundary(2,2):(DvBoundary(2,1)-DvBoundary(2,2))/N_show:DvBoundary(2,1)];

xshowflat=zeros(N_show+1,N_show+1);
yshowflat=zeros(N_show+1,N_show+1);
W_show=zeros(N_show+1,N_show+1,N_C);
p=0;
for i=1:N_show+1
    for j=1:N_show+1
        p=p+1;
        xshowflat(i,j)=xshow(i);
        yshowflat(i,j)=yshow(j);
        temp1_max=zeros(4,1);
        temp1=F.MLRmodel.pred([xshow(i), yshow(j)]);
        
        for k=1:N_C
        W_show(i,j,k)=temp1(k,1);
        end        
    end
end

figure()
set(gcf,'position',[200 200 1200 600]);
sgtitle('Contour of Model Weights','FontName', 'Times','fontweight','bold','FontSize',16)
for i=1:N_C
subplot(2,4,i)
hold on
box on
surf(xshowflat,yshowflat,W_show(:,:,i))
plot3(DV_org(:,1),DV_org(:,2),1*ones(N_doe,1),'ks','MarkerSize',11,'MarkerFaceColor','w')
plot3(F.X_sub{i}(:,1),F.X_sub{i}(:,2),1*ones(length(F.X_sub{i}(:,1)),1),'rs','MarkerSize',11,'MarkerFaceColor','r')
view(2)
shading interp

axis([DvBoundary(1,2) DvBoundary(1,1) DvBoundary(2,2) DvBoundary(2,1)])
xlabel('DV1')
ylabel('DV2')
end

%% Variance-based adaptive sampling
adjF=[10, 10, 1, 1, 0, 0]';
F=F.init_LOOCV(NRMSE, 0);
xnew_save=SLNIROM_AdaDoE(F,NRMSE, adjF);
%
figure()
set(gcf,'position',[500 200 550 480]);
set(gca,'FontName', 'Times','fontweight','bold','FontSize',18,'LineWidth',1.5);
set(gca,'XTick',[DvBoundary(1,2):(DvBoundary(1,1)-DvBoundary(1,2))/4:DvBoundary(1,1)])
set(gca,'YTick',[DvBoundary(2,2):(DvBoundary(2,1)-DvBoundary(2,2))/5:DvBoundary(2,1)])
set(gca,'XMinorTick','on','YMinorTick','on','TickLength',[0.02 0.02])


hold on
box on
M_size=15;
M_thickness=1.2;
plot(DV_org(:,1),DV_org(:,2),'ks','MarkerSize',11,'MarkerFaceColor','w')
plot(xnew_save(:,1),xnew_save(:,2),'ro','MarkerSize',11,'MarkerFaceColor','w')
view(2)
shading interp

axis([DvBoundary(1,2) DvBoundary(1,1) DvBoundary(2,2) DvBoundary(2,1)])
xlabel('DV1')
ylabel('DV2')
title('Variance-based Adaptive Sampling')

