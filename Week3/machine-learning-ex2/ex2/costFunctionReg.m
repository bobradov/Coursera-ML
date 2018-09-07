function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Compute standard cost and gradient first
% Then ammend with additional regularization terms

htheta = sigmoid( X*theta ); % matrix product for X theta
SumYEq1 = - y'*log(htheta);
SumYEq0 = -( ones(m,1)-y )' * log( ones(m,1)-htheta );

% Compute standard cost
J_standard = 1/m*( SumYEq1 + SumYEq0 );

% Compute standard gradient
grad_standard = 1/m * X'*( htheta - y );

% Now compute cost of regularization
theta_reg = theta; theta_reg(1) = 0.0;
%theta
%theta_reg
J_reg = lambda/(2*m)*theta_reg'*theta_reg;
J = J_standard + J_reg;
%disp(sprintf('J_standard=%g J_reg=%g J=%g', J_standard, J_reg, J));
%pause;

grad_reg = theta_reg*lambda/m;
grad = grad_standard + grad_reg;

% =============================================================

end
