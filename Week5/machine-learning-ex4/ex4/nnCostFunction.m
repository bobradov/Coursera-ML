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
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Feed-forward outputs
% Not prediction: not finding the max value to use as a classifier
% Only the floating-point output

% Augment the data set with a columnn of ones
Xf = [ ones(m,1), X];

% Compute the first layer activations
z2 = Theta1*Xf';
a2 = sigmoid( z2 );
%z1 = sigmoid( Theta1*Xf' );

%disp('size z1:');
%size(z1)

% Compute second layer activations
% Augment the 2nd-layer inputs with the column of ones for the bias
x2 = [ ones(m,1), a2'];

%disp('size x2:');
%size(x2)
z3 = Theta2*x2';
a3 = sigmoid( z3 );

%disp('size of output:');
%size(z2)

%disp('sample output:')
%z2(:,1)

% Now compute cost function, given the output and correct values
%disp('Size of y:');
%size(y)

% Loop over training exammples

% utility vectors that don't
% need to be re-generated in the loop
unity = ones(num_labels,1); 

% Compute cost of regularization
% Extract portion of Theta1, Theta2 which does not
% include the bias dependence
% We don't include the bias parameters in the regularization
% cost

Theta1SubSqr = Theta1(:,2:end).^2;
Theta2SubSqr = Theta2(:,2:end).^2;
RegCost1     = sum( sum( Theta1SubSqr ) );
RegCost2     = sum( sum( Theta2SubSqr ) );


for index = 1:m
	% Convert current training value into a vector that
	% can be compared to the output of the nn
	i = y(index);
	tempvec = zeros(num_labels,1);
	tempvec(i) = 1.0;
	

	% Now compute cost	
	costOne  = -tempvec'*log(a3(:,index));
	costZero = -(unity-tempvec)'*log(unity-a3(:,index));

	% Update total cost
	J = J + costOne + costZero; 
end

J = J/m + lambda*( RegCost1 + RegCost2 )/(2*m);

%size(J)



% Now compute gradients using back-propagation

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

%disp('Doing backprop ...');

for index = 1:m
	%disp(sprintf('Doing backprop for index=%d', index));

	i = y(index);
	tempy = zeros(num_labels,1);
	tempy(i) = 1.0;

	% Compute output error d3
	d3 = a3(:,index)-tempy;
	%d3

	% Compute d2 ("error" in layer2)
	%disp('Theta2');
	%size(Theta2')
	%disp('d3');
	%size(d3)
	%disp('z2(:,index)');
	%size(z2(:,index))

	d2_start = Theta2'*d3;
	%disp('d2_start')
	%size(d2_start)

	d2_next = d2_start(2:end);

	d2 = d2_next .* sigmoidGradient( z2(:,index) );
	%disp('d2')
	%size(d2)
	%d2

	% Accumulate Delta2
	% Augment activations to include the "+1" for the bias
	a2_temp = [ 1; a2(:,index)];

	%disp('Accumulating Delta2');
	%disp('d3')
	%size(d3)
	%disp('a2(:,index')
	%size(a2_temp)
	Delta2 = Delta2 + d3 * a2_temp';
	%size(Delta2)
	%disp('Expected size:')
	%size(Theta2_grad)

	% Accumulate Delta1

	a1_temp = Xf(index,:)';

	%disp('Accumulating Delta1');
	%disp('a1_temp')
	%size(a1_temp)
	%disp('d2')
	%size(d2)
	Delta1 = Delta1 + d2 * a1_temp';
	%size(Delta1)
	%disp('Expected size:')
	%size(Theta1_grad)

end
% -------------------------------------------------------------

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
