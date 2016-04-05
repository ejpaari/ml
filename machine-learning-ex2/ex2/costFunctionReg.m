function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples
h = sigmoid(X * theta);
theta(1) = 0;
J = 1/m * sum((-y.*log(h) - (1-y) .* log(1-h))) + lambda/(2*m) * sum(theta(2:end).^ 2);
grad = (1/m * X' * (h - y)) + lambda/m * theta;
end
