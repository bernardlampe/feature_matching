% draws a cross on the image
% assumes an RGB image

function Im = drawCross(Im, y, x, h, w, r, g, b)

% get image dimensions
[height width] = size(Im);

% draw the vertical line
for i = y - h: y + h
    if i > 0 && i <= height
        Im(i, x, 1) = r;
        Im(i, x, 2) = g;
        Im(i, x, 3) = b;
    end
end

% draw the vertical line
for i = x - w: x + w
    if i > 0 && i <= width
        Im(y, i, 1) = r;
        Im(y, i, 2) = g;
        Im(y, i, 3) = b;
    end
end