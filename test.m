function [cost, grad] = rica(W, layersizes, x)
  h = W*x;
  H = W'*h;
  a = H - x;
  cost = sum(a(:).^2);

  grad = 2 * W * (x * a' + a * x');

  % Compute a column of derivatives
  # for col = 1 : size(grad)(2),
  #   A = W * x(col);
  #   B = (2 * (H - x));
  #   C = 2 * (H(col) - x(col)) * (W * x);
  #   grad(:,col) = A * B + C;
  # end
end

%% [Input hidden]
layersizes = [3 4];

%% Weight initialization
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
    A = rand(layersizes(i+1), layersizes(i))*2*r - r; 
    theta(lold:lnew) = A(:);
end
theta = theta';

W = reshape(theta, layersizes(2), layersizes(1));
x = rand(layersizes(1),1);

[cost, grad] = rica(W, layersizes, x)

% [x, obj_value, convergence, iters] = bfgsmin(@rica, W, layersizes, x)

% derv = 2*(H(1) - x(1))*(2*x(1)*W(1,1)+W(1,2)*x(2)) + 2*(H(2)-x(2))*(W(1,2)*x(1))

% Compute an emperical derivate of the function
empGrad = zeros(layersizes(2), layersizes(1));
for row = 1 : size(grad)(1),
  for col = 1 : size(grad)(2),
    init = W(row,col);
    dx = 1;
    for i=1:10,
      W(row,col) = init + dx;
      h = W*x;
      H = W'*h;
      a = H - x;
      high = sum(a(:).^2);

      W(row,col) = init - dx;
      h = W*x;
      H = W'*h;
      a = H - x;
      low = sum(a(:).^2);

      newDerv = (high - low) / (2 * dx);
      dx = dx / 2;
    end
    empGrad(row,col) = newDerv;
    W(row,col) = init;
  end
end
empGrad


