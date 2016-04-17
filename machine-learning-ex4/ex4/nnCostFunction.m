function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         After implementing Part 2, you can check that your implementation 
%         is correct by running checkNNGradients
% Part 3: Implement regularization with the cost function and gradients.

Y = zeros(m, num_labels);
for i = 1:m
    Y(i,:) = 1:num_labels == y(i);
end

a1 = X;
z2 = [ones(m,1) a1] * Theta1';
a2 = sigmoid(z2);
z3 = [ones(m,1) a2] * Theta2';
a3 = sigmoid(z3);

J = 1/m * sum(sum(-Y.*(log(a3))-(1-Y) .* (log(1-a3))));
J = J + lambda / (2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

d3 = a3 - Y;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

for i = 1:m
  Theta1_grad = Theta1_grad + d2(i,:)' * [1 a1(i,:)];
  Theta2_grad = Theta2_grad + d3(i,:)' * [1 a2(i,:)];
end;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end);

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
