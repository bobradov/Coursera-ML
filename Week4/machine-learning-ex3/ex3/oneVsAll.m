function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

disp('Data dimensions before adding the column of ones:');
disp(sprintf('m=%d  n=%d num_labels=%d', m, n, num_labels));

disp('size y');
size(y)

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
%pause;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Train for nh-th digit


% Now train the logistic regression on this data subset


%     
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 

% Start with the '0' digit
c = 10;

labels = [ 1, 2, 3, 4, 5, ...
          6, 7, 8, 9, 10];

for index=1:num_labels

	c = labels(index);
	%disp(sprintf('Working on index=%d with label=%d', index, c));

	% reset Initial theta
	initial_theta = zeros(n + 1, 1);

	% Get initial error for digit
	%[jinit,gradinit] = lrCostFunction( initial_theta, X, (y == c), lambda);
	%disp(sprintf('For digit c=%d, Initial cost: %g', c, jinit));

	% Now perform optimization
	[theta] = ...
	         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
	                 initial_theta, options);


	% Store the result in 'all_theta'
	all_theta(index,:) = theta;         
end

%disp('Final theta:');
%theta

%pause;
%all_theta




% =========================================================================


end
