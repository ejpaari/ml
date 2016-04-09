function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y);
for iter = 1:num_iters
    h = X * theta;
    theta = theta - alpha * (1/m) * ((h-y)' * X)';
end
end
