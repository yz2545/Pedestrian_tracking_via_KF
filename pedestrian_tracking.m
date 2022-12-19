% Final project: pedestrian tracking usin Kalman Filter
% author: Yan Zhang

clear variables; close all; clc
%% add path
addpath(genpath('C:/Users/Yan''s laptop/Desktop/MAE 6760 Model-Based Estimation/Final project/code'))
%% load data
% east, north
load('carpose_imu_measurement.mat')
% x, y, z
load('pedestrian_lidar_measurement.mat')


%% state-space model and covariances
dt=1/10;%sec
F=[1 0 dt 0;
   0 1 0 dt;
   0 0 1 0 ;
   0 0 0 1];

H=[1 0 0 0;
   0 1 0 0];
ns=length(F);% number of states
% process noise
G=eye(4);
Q=diag([1 1 1 1]);
% sensor noise (lidar)
Rl=[1 0;
    0 1]; % for position x and y
% sensor noise (imu)
Ri=[1 0;
    0 1]*0.1; % for position x and y

%% Data association and estimation
% number of timesteps
nk=length(x); 
% timespan
t=(0:dt:dt*(nk-1));
% initialize the target list and target position
x0=cell2mat(x(1));
x0=x0(~isnan(x0));
y0=cell2mat(y(1));
y0=y0(~isnan(y0));
% estimate the initial velocity
% x1=cell2mat(x(2));
% x1=x1(~isnan(x1));
% y1=cell2mat(y(2));
% y1=y1(~isnan(y1));
% vx0=(x1-x0)/dt;
% vy0=(y1-y0)/dt;

% inital number of targets
n_target=length(x0);
% initial target list
% target_list=1:n_target;

% measurement gating 
Pgate=0.95;
nz=2; % # of DOF
Lam0=chi2inv(Pgate,nz);
% loop through all time steps
% nk=20 % for debugging
R=Rl+Ri;% account for both lidar and imu noise
P0=R;
% get initial condition for each detected target
xhatp=[x0; y0; zeros(1,n_target); zeros(1,n_target)];% 2 x ntarget
% xhatp=[x0;y0;vx0;vy0];
Pp(1:ns,1:ns,1:n_target,1:nk)=zeros(4,4,2,nk);
xhatu=[x0; y0; zeros(1,n_target); zeros(1,n_target)];% 2 x ntarget
% xhatu=[x0;y0;vx0;vy0];
Pu(1:ns,1:ns,1:n_target,1:nk)=zeros(4,4,2,nk);
for j=1:n_target
    Pp([1,2],[1,2],j,1)=P0;
    Pu([1,2],[1,2],j,1)=P0;
    % guess the variance of speed
    Pp([3,4],[3,4],j,1)=P0;
    Pu([3,4],[3,4],j,1)=P0;
end

% k:timestep, i:measurement, j:target
% xhatp_new=xhatp;% initialize
% count for # of no measurements
no_z_idx=zeros(1,n_target);
for k=1:(nk-1)
    
    for j=1:n_target
        % KF: predict
        if no_z_idx(j)>1
            xhatp(:,j,k+1)=nan(4,1);
            Pp(1:ns,1:ns,j,k+1)=nan(ns,ns);
        else
            xhatp(:,j,k+1) = F*xhatu(:,j,k);
            Pp(1:ns,1:ns,j,k+1) = F*Pu(1:ns,1:ns,j,k)*F' + G*Q*G';
            %Kalman Gain
            K(1:4,1:2,j) = Pp(1:ns,1:ns,j,k+1)*H'*inv(H*Pp(1:ns,1:ns,j,k+1)*H' + R);
        end
    end
    % get measurement
    x_vec=cell2mat(x(k+1));
    y_vec=cell2mat(y(k+1));
    % remove nan
    x_vec=x_vec(~isnan(x_vec));
    y_vec=y_vec(~isnan(y_vec));
%     % put the pedestrian's position in the same world coordinates as the carpose
%     x_vec=x_vec+east(k+1);
%     y_vec=y_vec+north(k+1);
    n_z=length(x_vec);
    z_target=nan(2,n_target);
    % data association
    for i=1:n_z
        dmin=Lam0;
        z=[x_vec(i); y_vec(i)];
        for j=1:n_target
            % innovation calculation
            inn=z-H*xhatp(:,j,k+1);
            S=H*Pp(1:ns,1:ns,j,k+1)*H'+R;
            % for the measurement gating test statistic
            Lam=inn'*inv(S)*inn;
            % Mahalanobis distance
            d(j)=sqrt(inn'*inv(S)*inn);
            if Lam<Lam0
                % if this measurement is smaller than the dmin, update dmin
                if d(j)<dmin
                    dmin=d(j);
                    z_target(:,j)=z;
                end
            end
        end
    end
    % Update
    for j=1:n_target
        % stop tracking if there are 2 measurements are missing
        if isnan(z_target(1,j))
            no_z_idx(j)=no_z_idx(j)+1;
            if no_z_idx(j)==1
                xhatu(:,j,k+1) = xhatp(:,j,k+1);
                Pu(1:ns,1:ns,j,k+1)=Pp(1:ns,1:ns,j,k+1);
            
            elseif no_z_idx(j)>1
                xhatu(:,j,k+1)=nan(ns,1);
                Pu(1:ns,1:ns,j,k+1)=nan(ns,ns);
            end
        else
            xhatu(:,j,k+1) = xhatp(:,j,k+1) + K(:,:,j) *(z_target(:,j) - H*xhatp(:,j,k+1));
            Pu(1:ns,1:ns,j,k+1) = (eye(ns)-K(:,:,j)*H)*Pp(1:ns,1:ns,j,k+1)*(eye(ns)-K(:,:,j)*H)' + K(:,:,j)*R*K(:,:,j)';
        end
    end
    
end

%% convert pedestrian location to the world coordinates
xhatu_w=xhatu;
% timestep k=1
theta=atan(east(2)/north(2));
xhatu_w(1:2,:,1)=funcIMU2world(squeeze(xhatu(1,:,1)),squeeze(xhatu(2,:,1)),east(1),north(1),theta);
% timesteps (2:nk)
for k=2:nk
   xhatu_w(1:2,:,k)=funcIMU2world(squeeze(xhatu(1,:,k)),squeeze(xhatu(2,:,k)),east(k),north(k),[]);
end 

%% plots
% in the world coordinates
plot_estimator(t,[],xhatu_w,Pu,1,1,"Target 1: East position");
saveas(gcf,'Fig_Target1_east_position.png')
plot_estimator(t,[],xhatu_w,Pu,1,2,"Target 2: East position");
saveas(gcf,'Fig_Target2_east_position.png')
plot_estimator(t,[],xhatu_w,Pu,2,1,"Target 1: North position");
saveas(gcf,'Fig_Target1_north_position.png')
plot_estimator(t,[],xhatu_w,Pu,2,2,"Target 2: North position");
saveas(gcf,'Fig_Target2_north_position.png')

plot_estimator(t,[],xhatu_w,Pu,3,1,"Target 1: East velocity");
saveas(gcf,'Fig_Target1_east_velocity.png')
plot_estimator(t,[],xhatu_w,Pu,3,2,"Target 2: East velocity");
saveas(gcf,'Fig_Target2_east_velocity.png')
plot_estimator(t,[],xhatu_w,Pu,4,1,"Target 1: North velocity");
saveas(gcf,'Fig_Target1_north_velocity.png')
plot_estimator(t,[],xhatu_w,Pu,4,2,"Target 2: North velocity");
saveas(gcf,'Fig_Target2_north_velocity.png')
% in the imu coordinates
% plot_estimator(t,[],xhatu,Pu,1,1,"Target 1: East position");
% plot_estimator(t,[],xhatu,Pu,1,2,"Target 2: East position");
% plot_estimator(t,[],xhatu,Pu,2,1,"Target 1: North position");
% plot_estimator(t,[],xhatu,Pu,2,2,"Target 2: North position");
% 
% plot_estimator(t,[],xhatu,Pu,3,1,"Target 1: East velocity");
% plot_estimator(t,[],xhatu,Pu,3,2,"Target 2: East velocity");
% plot_estimator(t,[],xhatu,Pu,4,1,"Target 1: North velocity");
% plot_estimator(t,[],xhatu,Pu,4,2,"Target 2: North velocity");

%% bird's eye view
figure(11)
plot(east,north,LineWidth=1.5)
hold on
plot(squeeze(xhatu_w(1,1,:)),squeeze(xhatu_w(2,1,:)),LineWidth=1.5)
plot(squeeze(xhatu_w(1,2,:)),squeeze(xhatu_w(2,2,:)),LineWidth=1.5)
% draw 2sigma error ellipse
timesteps=(1:10:60);
for i=1:length(timesteps)
    [Xe,Ye] = calculateEllipseCov(xhatu_w(1:2,1,timesteps(i)), Pu(1:2,1:2,1,timesteps(i)), 2);
    plot(Xe,Ye,'m--');
    [Xe,Ye] = calculateEllipseCov(xhatu_w(1:2,2,timesteps(i)), Pu(1:2,1:2,2,timesteps(i)), 2);
    plot(Xe,Ye,'m-');
end
hold off
xlabel('east [m]')
ylabel('north [m]')
legend('car','pedestrian #1','pedestrian #2','pedestrian #1:2\sigma bound','pedestrian #2: 2\sigma bound','Location','southwest')
saveas(gcf,'Fig_birdseyeview.png')
% figure(12)
% plot(squeeze(xhatu(1,1,:)),squeeze(xhatu(2,1,:)))
% hold on
% plot(squeeze(xhatu(1,2,:)),squeeze(xhatu(2,2,:)))

%% 
% plot_birdseyeview(Xnonoise,[],[],'Particle Distribution');
% plot_estimator_error_PF(t,Xnonoise(1,:),xEst(:,1)',xSig(:,1)',1,'PF: no bias estimation');
% % saveas(gcf,'Fig_hw5_carpose_x_PF_swervy.png')
% plot_estimator_error_PF(t,Xnonoise(2,:),xEst(:,2)',xSig(:,2)',2,'PF: no bias estimation');
% % saveas(gcf,'Fig_hw5_carpose_y_PF_swervy.png')
% plot_estimator_error_PF(t,Xnonoise(3,:),xEst(:,3)',xSig(:,3)',3,'PF: no bias estimation');
% % saveas(gcf,'Fig_hw5_carpose_vel_PF_swervy.png')
% plot_estimator_error_PF(t,Xnonoise(4,:),xEst(:,4)',xSig(:,4)',4,'PF: no bias estimation');
% % saveas(gcf,'Fig_hw5_carpose_heading_PF_swervy.png')

%% extrinsics function
% convert measurement from the IMU coordinates to the world coordinates
% i = timestep
function z_world=funcIMU2world(zx,zy,east,north,theta)
    if isempty(theta)
        theta=atan(east/north);
    end
    % rotation matrix
    Rot_imu=[cos(theta) -sin(theta);
        sin(theta) cos(theta)];
    % translation
    T_imu=[east;
           north];
    extrinsics=[Rot_imu T_imu;
                0 0 1];
    % convert pedestrian measurements to the world coordinate system
    z=extrinsics*[zx;zy;ones(1,length(zx))];
    z_world=z(1:2,:);
end

%% functions
function plot_estimator(t,x1,x2,P2,state_idx,target_idx, title_name);
% x1 is the true value or reference comparison
% x2,P2 is the estimator state and covariance
% ii_plot: 2x1 vector of which states to plot
%
axis_names={'East (m)','North (m)','East velocity (m/sec)','North velocity (m/sec)'};
figure;
ii_x1=[];ii_x2=[];ii_P2=[]; %for legend
%

hold on;
if ~isempty(x1),
    plot(t,x1(:,target_idx,:),'color',[0 0.8 0]);ii_x1=1;
end  
if ~isempty(x2),
    plot(t,squeeze(x2(state_idx,target_idx,:)),'b-');ii_x2=2;
end  
if ~isempty(P2)
    % xvar=variance
    xvar=squeeze(P2(state_idx,state_idx,target_idx,:));
    xbound=[[squeeze(x2(state_idx,target_idx,:))+2*sqrt(xvar)]', fliplr([squeeze(x2(state_idx,target_idx,:))-2*sqrt(xvar)]')];
    tbound=[t fliplr(t)];
    % remove the entries that are NaN
    keepIndex=~isnan(xbound) & ~isnan(tbound);
    xbound=xbound(keepIndex);
    tbound=tbound(keepIndex);
    patch(tbound,xbound,'b','EdgeColor','b','FaceAlpha',0.2,'EdgeAlpha',0.2);
    ii_P2=3;
end
hold off
xlabel('time (sec)');ylabel(axis_names(state_idx));grid;
% xlim([0 35]);set(gca,'xtick',[0:5:35]);

%
sgtitle(title_name);
PrepFigPresentation(gcf);
legend_names={'true state','estimate','2\sigma bound'};
legend(legend_names{ii_x1},legend_names{ii_x2},legend_names{ii_P2},'Location','South','fontsize',12);
end

function [Xe,Ye] = calculateEllipseCov(X, P, nsig, steps) 
    %# This functions returns points to draw an ellipse 
    %# 
    %#  @param X     x,y coordinates 
    %#  @param P     covariance matrix 
    %# 
 
    error(nargchk(2, 4, nargin)); 
    if nargin<3, nsig = 1; end 
    if nargin<4, steps = 36; end 
    
    [U,S,V]=svd(P);
    s1=sqrt(S(1,1));s2=sqrt(S(2,2));angle=acos(U(1,1))*180/pi;
    x=X(1);
    y=X(2);

    %scale by nsig
    s1=nsig*s1;
    s2=nsig*s2;

    beta = angle * (pi / 180); 
    sinbeta = sin(beta); 
    cosbeta = cos(beta); 
 
    alpha = linspace(0, 360, steps)' .* (pi / 180); 
    sinalpha = sin(alpha); 
    cosalpha = cos(alpha); 
 
    Xe = x + (s1 * cosalpha * cosbeta - s2 * sinalpha * sinbeta); 
    Ye = y + (s1 * cosalpha * sinbeta + s2 * sinalpha * cosbeta); 
 
end 

function PrepFigPresentation(fignum);
%
% prepares a figure for presentations
%
% Fontsize: 14
% Fontweight: bold
% LineWidth: 2
% 

figure(fignum);
fig_children=get(fignum,'children'); %find all sub-plots

for i=1:length(fig_children),
    
    set(fig_children(i),'FontSize',12);
    set(fig_children(i),'FontWeight','bold');
    
    fig_children_children=get(fig_children(i),'Children');
    set(fig_children_children,'LineWidth',2);
end
end