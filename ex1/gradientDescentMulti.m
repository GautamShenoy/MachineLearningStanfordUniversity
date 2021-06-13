function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%y = ( y - mean(y) ) / ( max(y) - min(y) );

for iter = 1:num_iters
    
    H_theta = X * theta;
    temp1Hyp = sum (H_theta - y );  
    temp2Hyp = sum( (H_theta - y) .* X(:, 2) );
    temp3Hyp = sum( (H_theta - y) .* X(:, 3) );
    temp1 = theta(1) - ( ( alpha / m ) * temp1Hyp );
    temp2 = theta(2) - ( ( alpha / m ) * temp2Hyp );
    temp3 = theta(3) - ( ( alpha / m ) * temp3Hyp );
    theta(1) = temp1;
    theta(2) = temp2;
    theta(3) = temp3;
    J_history(iter) = computeCostMulti(X, y, theta);
    
end

end
