function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos = [];    % Matrix that will contain data of positive training examples
neg = [];    % Matrix that will contain data of negative training examples
n = 1;
m = 1;
for i = 1:length(y)
	if (y(i) == 1)
		pos(n,:)= X(i,:);
		n = n+1;
	elseif (y(i) == 0)
		neg(m,:) = X(i,:);
		m = m+1;
	endif
end

plot(pos(:,1),pos(:,2),'k+','LineWidth',2,'MarkerSize',7);
plot(neg(:,1),neg(:,2),'ko','MarkerFaceColor','y','MarkerSize',7);

% =========================================================================



hold off;

end
