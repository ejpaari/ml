function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
for iter = 1:num_iters
    t1 = 0;
    t2 = 0;
        
    for i = 1:m
        t1 = t1 + (theta(1) + theta(2) * X(i,2) - y(i));
        t2 = t2 + ((theta(1) + theta(2) * X(i,2) - y(i)) * X(i,2));
    end
    
    theta(1) = (theta(1) - alpha * (1/m) * t1);
    theta(2) = (theta(2) - alpha * (1/m) * t2);
end
end
