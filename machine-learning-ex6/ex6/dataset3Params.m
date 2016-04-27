function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 0;
sigma = 0.0;

% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
a = [0.01 0.03 0.1 0.3 1 3 10 30];
minError = 999999999;

for i=1:length(a)
    for j=1:length(a)
        model = svmTrain(X, y, a(i), @(x1, x2) gaussianKernel(x1, x2, a(j)));
        predictions = svmPredict(model, Xval);
        predError = mean(double(predictions ~= yval));
        if predError < minError
            minError = predError;
            C = a(i);
            sigma = a(j);
        end
    end
end
end
