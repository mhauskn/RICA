function [] = main()
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
x = rand(layersizes(1),10);

[cost, grad] = rica(W, layersizes, x)
    
% Compute an emperical derivate of the function
% This is equivalent to going column-wise through 
% x and summing the derivatives
empGrad = zeros(layersizes(2), layersizes(1));
for row = 1 : size(grad,1),
  for col = 1 : size(grad,2),
    init = W(row,col);

    dx = 1e-10;
    W(row,col) = init + dx;
    h = W*x;
    H = W'*h;
    M = size(x,2);
    diff = H - x;
    high = 1/M * 0.5 * sum(diff(:).^2);
    s = log(cosh(h));
    sparsityCost = .5/M * sum(s(:));
    high = high + sparsityCost;

    W(row,col) = init - dx;
    h = W*x;
    H = W'*h;
    diff = H - x;
    low = 1/M * 0.5 * sum(diff(:).^2);
    s = log(cosh(h));
    sparsityCost = .5/M * sum(s(:));
    low = low + sparsityCost;

    newDerv = (high - low) / (2 * dx);
    dx = dx / 2;

    empGrad(row,col) = empGrad(row,col) + newDerv;
    W(row,col) = init;
  end
end
empGrad
grad ./ empGrad
end

function [cost, grad] = rica(W, layersizes, x)
  h = W*x;
  H = W'*h;
  M = size(x,2);
  diff = H - x;
  cost = 1/M * 0.5 * sum(diff(:).^2);
  
  s = log(cosh(h))
  sparsityCost = .5/M * sum(s(:));
  cost = cost + sparsityCost;

  grad = 1/M * W * (x * diff' + diff * x');
  
  tmp = tanh(h) * x';
  grad = grad + tmp;
end

main()