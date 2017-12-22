clear;
data = load('ex1data1.txt');
X = data(:,1);
Y = data(:,2);
m = length(X);
plotData(X,Y);

X=[ones(m,1), data(:,1)];

theta = zeros(2,1);



J = computeCost(X,Y,theta)
print J;

