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

X = [ones(m,1),X];       % Add bias unit to first layer
z2 = X*Theta1';
a2 = sigmoid(z2);        % a2: size m*25 ,hidden layer units
a2 = [ones(m,1),a2];     % Add bias unit to second layer 
z3 = a2*Theta2';         
h = sigmoid(z3);         % h: size m*num_labels,it has the value of our hypothesis

Y = zeros(m,num_labels);
for i=1:m
 	Y(i,y(i))=1;
end

J_main = -(1/m) * sum( sum ( Y.*log(h) + (ones(m,num_labels)-Y).*log(ones(m,num_labels)-h) ) );
J_reg = (lambda/(2*m))*( sum( sum( Theta1(:,2:end).^2 ) )+ sum( sum( Theta2(:,2:end).^2 ) ) );

J = J_main + J_reg;

D_2_main = zeros(num_labels,hidden_layer_size+1);
D_1_main = zeros(hidden_layer_size,input_layer_size+1);

for t=1:m
	
	a1_t = X(t,:);			
	z2_t = a1_t*Theta1';		
	a2_t = sigmoid(z2_t);
	a2_t = [1,a2_t];			
	z3_t = a2_t*Theta2';		
	h_t = sigmoid(z3_t);		

	y_t = zeros(1,num_labels);
	y_t(y(t)) = 1;
		
	d_3 = h_t - y_t;			
	d_2 = d_3*Theta2(:,2:end).*sigmoidGradient(z2_t);
	
	D_2_main = D_2_main + d_3'*a2_t;  
	D_1_main = D_1_main + d_2'*a1_t; 

end

D_2_reg = lambda.*[zeros(num_labels,1) Theta2(:,2:end)];
D_1_reg = lambda.*[zeros(hidden_layer_size,1),Theta1(:,2:end)];

D_2 = 1/m.*(D_2_main + D_2_reg);
D_1 = 1/m.*(D_1_main + D_1_reg);

Theta1_grad = D_1;
Theta2_grad = D_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
