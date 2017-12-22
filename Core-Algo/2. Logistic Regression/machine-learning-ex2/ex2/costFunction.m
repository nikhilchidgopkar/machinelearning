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

z = X*theta;
h = sigmoid(z);

logH = arrayfun(@log,h);
log1MinusH = arrayfun(@log,1-h);

t1 = (-1*y).*logH;
t2 = ((1-y)).*log1MinusH;

J = sum(t1-t2)/m;

%dif = transpose(X) * (h-y)

%sumCol = sum(transpose(X) * (h-y));

grad = transpose(transpose(h-y)*X/m);

print grad;

% =============================================================

end
