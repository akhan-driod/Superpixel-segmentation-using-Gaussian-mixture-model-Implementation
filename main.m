%%%%%%%%%%%%%%%%%%%%%
% script to train a GMM to segment apples in an image
%%%%%%%%%%%%%%%%%%%%%

function r=applesClassifier

close all
clear all

%load the apple images
run('LoadApplesScript.m');

%transform training images into apple/non-apple pixels
for i=1:3,
    %setting variables
    im = apples{i};
    mask = applesMasks{i};
    mask(mask == 255) = 1;
    im_width = size(im,1);
    im_height = size(im,2);
    im_channel = 3;
    
    im_apple = im .* double(mask);
    im_apple=reshape(permute(im_apple,[3,1,2]),[im_channel,im_width*im_height]);
    im_apple(:,all(im_apple==0))=[];
    
    im_non_apple = im .* double(not(mask));
    im_non_apple=reshape(permute(im_non_apple,[3,1,2]),[im_channel,im_width*im_height]);
    im_non_apple(:,all(im_non_apple==0))=[];
    
    if exist('px_apple','var') == 0
        px_apple = im_apple;
        px_non_apple = im_non_apple;
    else
        px_apple = cat(2,px_apple,im_apple);
        px_non_apple = cat(2,px_non_apple,im_non_apple);
    end;
end;

%%%%%%%%%%%%
% train the GMM
%%%%%%%%%%%5

% select a sizeable / random select of the training pixels
% this achieves a similar result to selecting all pixels with much reduced training time
training_size = 10000;
px_apple = px_apple(:,randperm(training_size));
px_non_apple = px_non_apple(:,randperm(training_size));

% learning the probability distributions for apple/non-apple pixels
% k was found empirically
k = 5;
apple_mixGauss = fitMixGauss(px_apple,k);
non_apple_mixGauss = fitMixGauss(px_non_apple,k);


%%%%%%%%%%%
% testing the GMM
%%%%%%%%%%%

% define priors of apples/non-apples
% roughly the proportion of the test images taken up by apples
priorApple = 0.5;
priorNonApple = 0.5;

% select and transform the test image
im_test = test_apples{4};
im_test_width = size(im_test,1);
im_test_height = size(im_test,2);
im_test_channel = 3;

im_test=reshape(permute(im_test,[3,1,2]),[im_test_channel,im_test_width*im_test_height]);

% loop through all the pixels in the test image
% assign a posterior probability to each
nData = im_test_width * im_test_height;
posteriorApple = zeros(1,nData);  
for (i = 1:nData)
    %extract this pixel data
    thisPixelData = double(im_test(:,i));
    %calculate likelihood of this data given skin model
    MLApple = zeros(k,1);
    MLNonApple = zeros(k,1);
    for cGauss = 1:k,
        MLApple(cGauss) = getGaussProb(thisPixelData,apple_mixGauss.mean(:,cGauss),apple_mixGauss.cov(:,:,cGauss));
        MLNonApple(cGauss) = getGaussProb(thisPixelData,non_apple_mixGauss.mean(:,cGauss),non_apple_mixGauss.cov(:,:,cGauss));
    end;
    %calculate likelihood of this data given apple model
    likeApple = apple_mixGauss.weight * MLApple;
    %calculate likelihood of this data given non apple model
    likeNonApple = non_apple_mixGauss.weight * MLNonApple;
    %calculate posterior probability from likelihoods and 
    %priors using BAYES rule
    pApple = likeApple * priorApple;
    pNApple = likeNonApple * priorNonApple;
    posteriorApple(i) = pApple / (pApple + pNApple);        

end;

%%%%%%%%%%%%%%%%%%%
% visualising results
%%%%%%%%%%%%%%%%%%

%plotting the ROC curve
mask_true = imread('testApples/internet_1_mask.png');
mask_true = mask_true(:,:,1); %only select the first 'color' channel
mask_true = mask_true(:);
mask_true(mask_true >0) = 1;
mask_predict = posteriorApple(:);

[X,Y,T,AUC] = perfcurve(mask_true,mask_predict,1);
AUC
plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Apple Image')

%visualising image
im_display = permute(reshape(im_test,[im_test_channel,im_test_width,im_test_height]),[2,3,1]);
mask_true = permute(reshape(mask_true,[1,im_test_width,im_test_height]),[2,3,1]);
mask_display = permute(reshape(posteriorApple,[1,im_test_width,im_test_height]),[2,3,1]);

%display test image and ground truth;
close all;
figure; set(gcf,'Color',[1 1 1]);
subplot(1,3,1); imagesc(im_display); axis off; axis image;
subplot(1,3,2); imagesc(mask_true); axis off; axis image;
clims = [0, 1];
subplot(1,3,3); imagesc(mask_display, clims); colormap(gray); axis off; axis image;
%drawnow;
a=3;

%==========================================================================
%==========================================================================


%%%%%%%%%%%%%%%%%%%
% functions to help with the script above
%%%%%%%%%%%%%%%%%%%5

%function to train the Gaussian parameters
function mixGaussEst = fitMixGauss(data,k);
        
[nDim nData] = size(data);

%MAIN E-M ROUTINE

% initialise the latent variables
% used to assign a responsibility to Gaussian model
postHidden = zeros(k, nData);

%in the E-M algorithm, we calculate a complete posterior distribution over
%the (nData) hidden variables in the E-Step.  In the M-Step, we
%update the parameters of the Gaussians (mean, cov, w).  

%initialize the values to random values
mixGaussEst.d = nDim;
mixGaussEst.k = k;
mixGaussEst.weight = (1/k)*ones(1,k);
mixGaussEst.mean = 2*randn(nDim,k);
for (cGauss =1:k),
    mixGaussEst.cov(:,:,cGauss) = (0.5+1.5*rand(1))*eye(nDim,nDim);
end;

%calculate current likelihood
logLike = getMixGaussLogLike(data,mixGaussEst,k);
fprintf('Log Likelihood Iter 0 : %4.3f\n',logLike);

nIter = 20;
for cIter = 1:nIter
    
   %Expectation step
   for (cData = 1:nData)
        for cGauss = 1:k,
            noms(cGauss) = mixGaussEst.weight(cGauss) * getGaussProb(data(:,cData),mixGaussEst.mean(:,cGauss),mixGaussEst.cov(:,:,cGauss));
        end;
        
        denominator = sum(noms);
        postHidden(:,cData) = noms / denominator;
       
   end;
   
   %Maximization Step
   for (cGauss = 1:k) 
        mixGaussEst.weight(cGauss) = sum(postHidden(cGauss,:)) / nData; 
        resp_sum = sum(postHidden(cGauss,:));
        mixGaussEst.mean(:,cGauss) = sum(bsxfun(@times,postHidden(cGauss,:),data),2) / resp_sum;
       
	% update parameters
        m_gauss = mixGaussEst.mean(:,cGauss);
        cov_values = zeros(nDim,nDim);
        for i = 1:nData,
            diff_1 = (data(:,i) - m_gauss);
            diff_2 = diff_1 * diff_1';
            cov_values = cov_values + postHidden(cGauss,i) * diff_2;
        end;
        
        mixGaussEst.cov(:,:,cGauss) = cov_values / resp_sum;

        
        
   end;
 

   %calculate the log likelihood
   logLike = getMixGaussLogLike(data,mixGaussEst,k);

end;

% ======================================

%calculate the gaussian probability for given data
function like = getGaussProb(data,gaussMean,gaussCov)

diff = data - gaussMean;
like = (1 / ((2*pi)^1.5*(det(gaussCov))^0.5)) * exp(-0.5*(data - gaussMean)'*inv(gaussCov)*(data - gaussMean));

% ==================================

%the goal of this routine is to calculate the log likelihood for the whole
%data set under a mixture of Gaussians model. We calculate the log as the
%likelihood will probably be a very small number that Matlab may not be
%able to represent.
function logLike = getMixGaussLogLike(data,mixGaussEst,k);

%find total number of data items
nData = size(data,2);

%initialize log likelihoods
logLike = 0;

%run through each data item
for(cData = 1:nData)
    thisData = data(:,cData); 
    
    weight = mixGaussEst.weight;
    mean = mixGaussEst.mean;
    cov = mixGaussEst.cov;
    
    like = zeros(k,1);
    for cGauss = 1:k,
        like(k) = weight(cGauss) * getGaussProb(thisData,mean(:,cGauss),cov(:,:,cGauss));
    end;
    
    %add to total log like
    logLike = logLike+log(sum(like));
end;


%The goal fo this routine is to draw the data in histogram form and plot
%the mixtures of Gaussian model on top of it.
function r = drawEMData2d(data,mixGauss)


set(gcf,'Color',[1 1 1]);
plot(data(1,:),data(2,:),'k.');

for (cGauss = 1:mixGauss.k)
    drawGaussianOutline(mixGauss.mean(:,cGauss),mixGauss.cov(:,:,cGauss),mixGauss.weight(cGauss));
    hold on;
end;
plot(data(1,:),data(2,:),'k.');
axis square;axis equal;
axis off;
hold off;drawnow;

    


%=================================================================== 
%===================================================================

%draw 2DGaussian
function r= drawGaussianOutline(m,s,w)

hold on;
angleInc = 0.1;

c = [0.9*(1-w) 0.9*(1-w) 0.9*(1-w)];


for (cAngle = 0:angleInc:2*pi)
    angle1 = cAngle;
    angle2 = cAngle+angleInc;
    [x1 y1] = getGaussian2SD(m,s,angle1);
    [x2 y2] = getGaussian2SD(m,s,angle2);
    plot([x1 x2],[y1 y2],'k-','LineWidth',2,'Color',c);
end

%===================================================================
%===================================================================

%find position of in xy co-ordinates at 2SD out for a certain angle
function [x,y]= getGaussian2SD(m,s,angle1)

if (size(s,2)==1)
    s = diag(s);
end;

vec = [cos(angle1) sin(angle1)];
factor = 4/(vec*inv(s)*vec');

x = cos(angle1) *sqrt(factor);
y = sin(angle1) *sqrt(factor);

x = x+m(1);
y = y+m(2);
