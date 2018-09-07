function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for jind = 1:K
	%disp('idx')
	% Find all x values that belong to index jind
	rows = find( idx==jind );
	[ njind, dummy ] = size(rows);
	%disp('rows:')
	%size(rows);

	xValsForj = X(rows,:);
	%disp('xValsForj');
	%size(xValsForj)
	%pause;
	% Sum across first dimension (rows)
	% Divide by number of data points in this cluster
	centroids( jind, : ) = sum( xValsForj, 1 )/njind;
	%disp('NewMean');
	%size(NewMean)
	%centroids( jind, : ) = NewMean;
end







% =============================================================


end

