function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Amend the data set with a columnn of ones

Xf = [ ones(m,1), X];

% Compute the first layer activations

z1 = sigmoid( Theta1*Xf' );

disp('size z1:');
size(z1)

x2 = [ ones(m,1), z1'];

disp('size x2:');
size(x2)

z2 = sigmoid( Theta2*x2' );

%disp('1');
%z2(:,1)
%disp('501');
%z2(:,501)
%disp('1001')
%z2(:,1001)

[pvals, pp] =  max( z2, [], 1 );
size(pp)

p = pp';

%p(1)
%p(501)
%p(1001)
%p(1501)





% ========================================================================%