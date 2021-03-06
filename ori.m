%カレントディレクトリ上の全てのjpgファイルを取得するため、余計なファイルはおかない！！

clear all;
close all;
addpath subfunctions
addpath images


fileList = dir('**/*.jpg');
%list = dir(append(folder,'*.txt'));

%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% １つ１つファイルを処理していく
for n = 1:length(fileList)
    % ファイルのフルパスを取得
    
    %disp(fileList(n).folder);
    %disp(fileList(n).name);
    fullpath = fullfile(fileList(n).folder, fileList(n).name);
    % メインの処理
    %disp(fileList(n));
    %disp(fullpath)
%disp(fullpath);

imname = fullpath;
u_org = double(imread(imname))/255;
[n1,n2,n3] = size(u_org);
N = n1*n2*n3;

order = 1; % 1: DTV, 2:DTGV
w1 = 0.5; % weight of luminance variation
 w2 = 0.5; % weight of second-order luminance variation for DVTGV
 alpha = [0.5, 0.5]; % parameter of DVTGV

%---------------------------------------------------------------
% comment out one 'problemtype' and the corresponding parameters

% problemtype = 'Denoise';
% sigma = 0.09; % noise standard deviation (normalized)
% tau = 0.95; % fidelity parameter

% problemtype = 'Deblur';
% sigma = 25.5/255; % noise standard deviation (normalized)
% tau = 1; % fidelity parameter
% psfsize = 5; % size of PSF
% psfstd = 2; % standard deviation of Gaussian blur

problemtype = 'MissingRecover';
missrate = 0.7; % percentage of missing components

% problemtype = 'DetailMag';
% epsilon = sqrt(N*0.01); % parameter of L2-ball
% beta = 5; % magnification rate
% w1 = 0.01; % weight of luminance variation
%---------------------------------------------------------------

stopcri = 1e-2; % stopping criterion
maxiter = 2000; % maximum number of iteration
gamma1 = 0.01; % parameter of PDS
gamma2 = 1/(12*gamma1); % parameter of PDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% observation generation

switch problemtype
    case 'Denoise'
        P = @(z) z;
        Pt = @(z) z;
 %       u_obsv = u_org + sigma*randn(size(u_org));
        u_obsv = u_org ;
        epsilon = tau*sqrt((sigma^2)*N);
    case 'Deblur'
        blu = zeros(n1, n2);
        blu(1:psfsize, 1:psfsize) = fspecial('gaussian', [psfsize psfsize], psfstd);
        blu = circshift(blu, [-(psfsize-1)/2 -(psfsize-1)/2]);
        h = fft2(blu);
        ht = conj(h);
        h = repmat(h, [1 1 n3]);
        ht = repmat(ht, [1 1 n3]);
        P = @(z) real(ifft2((fft2(z)).*h));
        Pt = @(z) real(ifft2((fft2(z)).*ht));
        u_obsv_org = P(u_org);
        u_obsv = u_obsv_org + sigma*randn(size(u_obsv_org));
        epsilon = tau*sqrt((sigma^2)*N);
    case 'MissingRecover'
        dr = randperm(N)';
        mesnum =round(N*(1-missrate));
        OM = dr(1:mesnum);
        OMind = zeros(n1,n2,n3);
        OMind(OM) = 1;
        P = @(z) z.*OMind;
        Pt = @(z) z.*OMind;
        u_obsv = P(u_org);
        epsilon = 0;
    case 'DetailMag'
        P = @(z) z;
        Pt = @(z) z;
        u_obsv = u_org;
end

psnrInput = EvalImgQuality(u_obsv, u_org, 'PSNR');
disp(['Input PSNR = ', num2str(psnrInput)]);
deltaEInput = EvalImgQuality(u_obsv, u_org, 'Delta2000');
disp(['Input deltaE = ', num2str(deltaEInput)]);

%% definitions for algorithm

% difference operators
D = @(z) cat(4, z([2:n1, n1],:,:) - z, z(:,[2:n2, n2],:)-z);
Dt = @(z) [-z(1,:,:,1); - z(2:n1-1,:,:,1) + z(1:n1-2,:,:,1); z(n1-1,:,:,1)] ...
    +[-z(:,1,:,2), - z(:,2:n2-1,:,2) + z(:,1:n2-2,:,2), z(:,n2-1,:,2)];

if order == 2;
    Dx = @(z) z([2:n1, n1],:,:) - z;
    Dy = @(z) z(:,[2:n2, n2],:)-z;
    Dxt = @(z) [-z(1,:,:); - z(2:n1-1,:,:) + z(1:n1-2,:,:); z(n1-1,:,:)];
    Dyt = @(z) [-z(:,1,:), - z(:,2:n2-1,:) + z(:,1:n2-2,:), z(:,n2-1,:)];
    G = @(z) cat(4, -Dxt(z(:,:,:,1)), -Dyt(z(:,:,:,1)) - Dxt(z(:,:,:,2)), -Dyt(z(:,:,:,2)));
    Gt = @(z) cat(4, -Dx(z(:,:,:,1)) - Dy(z(:,:,:,2)), -Dx(z(:,:,:,2)) - Dy(z(:,:,:,3)));
end

% color transform
C = @(z) cat(3, sum(z,3)*(sqrt(3)^(-1)), (z(:,:,1) - z(:,:,3))*(sqrt(2)^(-1)),...
    (z(:,:,1) - 2*z(:,:,2) + z(:,:,3))*(sqrt(6)^(-1)));
Ct = @(z) cat(3, (1/sqrt(3))*z(:,:,1) + (1/sqrt(6))*z(:,:,3) + (1/sqrt(2))*z(:,:,2),...
    (1/sqrt(3))*z(:,:,1) - (2/sqrt(6))*z(:,:,3), (1/sqrt(3))*z(:,:,1) + (1/sqrt(6))*z(:,:,3) - (1/sqrt(2))*z(:,:,2));

% L, prox_f1, and prox_f2
x{1} = u_obsv;
switch order
    case 1
        L = @(z) {D(C(z{1})), P(z{1})};
        Lt = @(z) {Ct(Dt(z{1})) + Pt(z{2})};
        
        prox_f1{1} = @(z, gamma) ProjDynamicRangeConstraint(z, [0,1]);
        prox_f2{1} = @(z, gamma) ProxDVTVnorm(z, gamma, w1);
        prox_f2{2} = @(z, gamma) ProjL2ball(z, u_obsv, epsilon);
    case 2
        L = @(z) {D(C(z{1})) - z{2}, G(z{2}), P(z{1})};
        Lt = @(z) {Ct(Dt(z{1})) + Pt(z{3}), -z{1} + Gt(z{2})};
        
        prox_f1{1} = @(z, gamma) ProjDynamicRangeConstraint(z, [0,1]);
        prox_f1{2} = @(z, gamma) z;
        prox_f2{1} = @(z, gamma) ProxDVTVnorm(z, alpha(1)*gamma, w1);
        prox_f2{2} = @(z, gamma) ProxDVTVnorm(z, alpha(2)*gamma, w2);
        prox_f2{3} = @(z, gamma) ProjL2ball(z, u_obsv, epsilon);
        
        x{2} = D(C(x{1}));
end

y = L(x);
xnum = numel(x);
ynum = numel(y);

%% main loop
for i = 1:maxiter
    % primal update
    xpre = x;
    x = cellfun(@(z1, z2) z1 - gamma1*z2, x, Lt(y), 'UniformOutput', false);
    for j = 1:xnum
        x{j} = prox_f1{j}(x{j}, gamma1);
    end
    
    % dual update
    Ltemp = L(cellfun(@(z1,z2) 2*z1 - z2, x, xpre, 'UniformOutput', false));
    for j = 1:ynum;
        Ltemp{j} = y{j} + gamma2 * Ltemp{j};
        y{j} = Ltemp{j} - gamma2 * prox_f2{j}(Ltemp{j}/gamma2, 1/gamma2);
    end
    
    error = sqrt(sum(sum(sum((xpre{1} - x{1}).^2))));
    if error < stopcri
        break;
    end
end
u_res = x{1}; % resulting image

%% result plot

psnrInput = EvalImgQuality(u_res, u_org, 'PSNR');
%disp(['Output PSNR = ', num2str(psnrInput)]);
deltaEInput = EvalImgQuality(u_res, u_org, 'Delta2000');
%disp(['Output deltaE = ', num2str(deltaEInput)]);
%imshow(u_res)

%disp(fileList(n).folder);
    %disp(fileList(n).name);

%ファイルの保存
filename=append("CS_",fileList(n).name);
path = fullfile(fileList(n).folder, filename);
disp(path)
imwrite(u_res,path)

end











