function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
for iter = 1:num_iters
    h = X * theta;
    theta = theta - alpha * (1/m) * ((h-y)' * X)';
end
end
