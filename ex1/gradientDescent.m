function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters
    H_theta = X * theta;
    temp1Hyp = sum (H_theta - y );  
    temp2Hyp = sum( (H_theta - y) .* X(:, 2) );
    temp1 = theta(1) - ( ( alpha / m ) * temp1Hyp );
    temp2 = theta(2) - ( ( alpha / m ) * temp2Hyp );
    theta(1) = temp1;
    theta(2) = temp2;

end

end
