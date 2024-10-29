% This code is for question2 ————————dejin wang

% Generate training and validation data
Ntrain = 100;
Nvalidate = 1000;
[xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate);

% Compute ML estimator
% Assume y = c(x, w) is a cubic polynomial in x
% xTrain is a 2xN matrix, so we need to construct a matrix with full cubic polynomial features

% Constructing the full cubic polynomial features for training data
Xtrain_poly = [ones(1, Ntrain);                % Constant term
               xTrain(1,:);                    % x1
               xTrain(2,:);                    % x2
               xTrain(1,:).^2;                 % x1^2
               xTrain(1,:) .* xTrain(2,:);      % x1 * x2
               xTrain(2,:).^2;                 % x2^2
               xTrain(1,:).^3;                 % x1^3
               xTrain(1,:).^2 .* xTrain(2,:);   % x1^2 * x2
               xTrain(1,:) .* xTrain(2,:).^2;   % x1 * x2^2
               xTrain(2,:).^3];                % x2^3

% ML estimate for w (least squares)
w_ML = (Xtrain_poly * Xtrain_poly') \ (Xtrain_poly * yTrain');

% Constructing the full cubic polynomial features for validation data
Xvalidate_poly = [ones(1, Nvalidate);           % Constant term
                  xValidate(1,:);               % x1
                  xValidate(2,:);               % x2
                  xValidate(1,:).^2;            % x1^2
                  xValidate(1,:) .* xValidate(2,:); % x1 * x2
                  xValidate(2,:).^2;            % x2^2
                  xValidate(1,:).^3;            % x1^3
                  xValidate(1,:).^2 .* xValidate(2,:); % x1^2 * x2
                  xValidate(1,:) .* xValidate(2,:).^2; % x1 * x2^2
                  xValidate(2,:).^3];           % x2^3

% Predict on the validation set
yPred_ML = w_ML' * Xvalidate_poly;

% Calculate the average squared error on the validation set (MSE)
error_ML = mean((yValidate - yPred_ML).^2);
fprintf('ML Average Squared Error: %.4f\n', error_ML);

% Compute MAP estimator
% Calculate Ridge regression (MAP) for different values of gamma
gammas = logspace(-3, 3, 10); % gamma from 10^-3 to 10^3
error_MAP = zeros(length(gammas), 1);

for i = 1:length(gammas)
    gamma = gammas(i);
    % MAP estimate for w (Ridge regression with regularization)
    w_MAP = (Xtrain_poly * Xtrain_poly' + 1/gamma * eye(size(Xtrain_poly, 1))) \ (Xtrain_poly * yTrain');
    
    % Predict on the validation set
    yPred_MAP = w_MAP' * Xvalidate_poly;
    
    % Calculate average squared error (MSE) on the validation set
    error_MAP(i) = mean((yValidate - yPred_MAP).^2);
end

% Visualize the average squared error for MAP estimator as gamma varies
figure;
semilogx(gammas, error_MAP, '-o');
xlabel('\gamma (Gamma)');
ylabel('Average Squared Error');
title('Average Squared Error of MAP Estimator with Varying \gamma');
