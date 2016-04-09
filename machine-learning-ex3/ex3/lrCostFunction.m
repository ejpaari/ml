function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

h = sigmoid(X * theta);
theta(1) = 0;
J = 1/m * sum(-y.*(log(h))-(1-y) .* (log(1-h))) + lambda/(2*m) * sum(theta.^ 2);
grad = (1/m * X' * (h - y)) + lambda/m * theta;

grad = grad(:);

end
