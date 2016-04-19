function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

h = X * theta;
theta(1) = 0;
J = 1/(2*m) * sum((h-y).^2) + lambda/(2*m) * sum(theta.^2);
grad = (1/m * X'*(h-y)) + lambda/m * theta;

end
