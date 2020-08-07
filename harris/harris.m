% Bernard Lampe
% Homework #4

% Harris corner detection algorithm
% EECS 442, University of Michigan, Fall 2009
% Algorithm outlined in lecture 16, slide 51

% <arg1> = filename of the image
% <arg2> = std. dev. of the gaussian filter
% <arg3> = window size to compute the 2nd moment matricies
% <arg4> = threshold for corner detection [0,1]

function harris(fname, sigma, winSize, threshold)

% empirical constant per lecture 16, slide 49
k = 0.04;

% check the arguments
if nargin ~= 4
    error('Usage: harris( <image filename> <sigma> <window size> <threshold>)');
end

% read in image file
Im = imread(fname);

% convert to grayscale
info = imfinfo(fname);
if strcmp(info.ColorType, 'grayscale')
    Im_gray = Im;
else
    Im_gray = rgb2gray(Im);
end

% cast to double for computation
Im_d = double(Im_gray);

% construct gaussian filter, use std. dev. passed in and
% make kernel size large enough to hold 3 standard deviations
ksize = 1 + 2 * ceil(2.5 * sigma);
gaus = fspecial('gaussian', ksize, sigma);

% convolve 2D image with gaussian filter
Im_gaus = imfilter(Im_d, gaus, 'symmetric', 'conv');

% compute the magnitude of the derivatives in the x direction
xfilt = [-1 0 1; -1 0 1; -1 0 1];
Im_dx = imfilter(Im_gaus, xfilt, 'replicate', 'conv');

% compute the magnitude of the derivatives in the y direction
yfilt = xfilt';
Im_dy = imfilter(Im_gaus, yfilt, 'replicate', 'conv');

% create squared derivative images (dx^2, dy^2)
Im_dx_2 = Im_dx.^2;
Im_dy_2 = Im_dy.^2;

% create image of cross derviaties
Im_dx_dy = Im_dx .* Im_dy;

% sum the entries about the window in each image using convolution
% use a gaussian window weighting with 3 std. devs. in the window
s = winSize / 5;
win = fspecial('gaussian', winSize, s);
Im_dx_2_sum = imfilter(Im_dx_2, win, 'replicate', 'conv');
Im_dy_2_sum = imfilter(Im_dy_2, win, 'replicate', 'conv');
Im_dx_dy_sum = imfilter(Im_dx_dy, win, 'replicate', 'conv');

% construct second moment matricies
M = zeros([size(Im_gray), 2, 2]);
[height, width] = size(Im_gray);
for h = 1:height
    for w = 1:width
        M(h, w, 1, 1) = Im_dx_2_sum(h, w);
        M(h, w, 2, 2) = Im_dy_2_sum(h, w);
        M(h, w, 1, 2) = Im_dx_dy_sum(h, w);
        M(h, w, 2, 1) = Im_dx_dy_sum(h, w);
    end
end

% compute the corner response values R = det(M(i,j)) - k * (trace(M(i,j))^2
% determinant M = eig(1) * eig(2)
% trace M = eig(1) + eig(2)
R = zeros(size(Im_gray));
for h = 1:height
    for w = 1:width
        m = reshape(M(h, w, :, :), 2, 2);
        e = eig(m);
        detM = e(1) * e(2); % compute determinant
        traceM = e(1) + e(2); % compute trace
        R(h, w) = detM - k * traceM * traceM; % compute corner response
    end
end

% threshold R to compute corners
rmin = min(R);
rmax = max(R);
Im_th = zeros(height, width);
for h = 1:height
    for w = 1:width
        if R(h, w) > (rmax - rmin) * threshold + rmin;
            Im_th(h, w) = R(h, w);
        end
    end
end

% find the image region peaks
Im_th = imregionalmax(Im_th);

% draw ellipses of the 2nd moment matricies on the image
figure;
imagesc(Im_gray);
colormap('gray');
hold on;
for h = 1:height/2
    for w = 1:width/2
        hp = 2 * h;
        wp = 2 * w;
        
        % get eigenvectors and values for this matrix
        [V, D] = eig(reshape(M(hp, wp, :, :), 2, 2));
        
        % only draw the regions with significant gradient
        eps = 1e-5;
        if ~(D(1,1) < eps || D(2,2) < eps)
            W = V * 1/sqrt(D);
            t = linspace(0, 2 * pi, 50);
            plot(W(1,1) .* cos(t) + W(1,2) .* sin(t) + wp, ...
                 W(2,1) .* cos(t) + W(2,2) .* sin(t) + hp, 'm');
        end
    end
end

% draw a green colored cross at each location of a detection on the
% original image
for h = 1:height
    for w = 1:width
        if Im_th(h, w) > 0
            Im = drawCross(Im, h, w, 3, 3, 0, 255, 0);
        end
    end
end

% display annotated image 
figure;
imagesc(Im);
