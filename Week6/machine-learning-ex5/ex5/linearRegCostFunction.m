function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Compute cost function
h = X*theta;
Delta = h-y;
DeltaSqr = Delta'*Delta;

ThetaForReg = theta(2:end,1);
RegTerm     = ThetaForReg'*ThetaForReg;

% Regularized cost function J
J = (1/(2*m))*DeltaSqr + lambda * RegTerm / (2*m);

% Compute gradient

% Un-regularized part
gradUnReg = (1/m) * Delta'*X;
%disp('gradUnReg');
%size(gradUnReg)
%pause;

% Add regularization
ThetaForRegGrad = theta;
ThetaForRegGrad(1,1) = 0.0;

grad = gradUnReg + (lambda/m) * ThetaForRegGrad';
%disp('grad')
%size(grad)
%pause; 











% =========================================================================

grad = grad(:);

end
