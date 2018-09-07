function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
	%size(theta)
	%size(X)
	%size(y)

	htheta = sigmoid( X*theta ); % matrix product for X theta
	%disp('htheta');
	%size(htheta)

	SumYEq1 = - y'*log(htheta);
	%disp(sprintf('Sum1=%g', SumYEq1 ) );

	SumYEq0 = -( ones(m,1)-y )' * log( ones(m,1)-htheta );
	%disp(sprintf('Sum0=%g', SumYEq0 ) );


	J = 1/m*( SumYEq1 + SumYEq0 );
	%disp(sprintf('J=%g', J));

	% Compute gradient
	grad = 1/m * X'*( htheta - y );
	%disp('grad:');
	%size(grad)




% =============================================================

end
