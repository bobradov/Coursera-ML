function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
	Z = zeros(size(X, 1), K);
	%disp('Requested Z:');
	%size(Z)
	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the projection of the data using only the top K 
	%               eigenvectors in U (first K columns). 
	%               For the i-th example X(i,:), the projection on to the k-th 
	%               eigenvector is given as follows:
	%                    x = X(i, :)';
	%                    projection_k = x' * U(:, k);
	%
	% Transpose of the U matrix
	% Columns of Utran are projections of the eigenvectors onto
	% the 'x' basis
	Utran = U';

	% Reduced order U matrix: only the first K rows 
	Utran_red = Utran(1:K,:);
	%disp('X:');
	%size(X)

	PsiU = Utran_red * X';
	%disp('PsiU');
	%size(PsiU)

	Z = PsiU';
	%disp('Z final:')
	%size(Z)

	pause;

%



% =============================================================

end
