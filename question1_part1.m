% Define parameters
% This code is for question1 part 1————————dejin wang
clear
clc

% Class priors
prior_0 = 0.6;
prior_1 = 0.4;

% Weights for class-conditional Gaussians
w = [0.5, 0.5]; 

% Mean vectors for class 0 and class 1
m01 = [-0.9, -1.1];
m02 = [0.8, 0.75];
m11 = [-1.1, 0.9];
m12 = [0.9, -0.75];

% Covariance matrix for all Gaussian distributions
Cij = [0.75, 0; 0, 1.25];

% Total number of samples
n_samples = 10000;

% Generate data
rng('default'); % For reproducibility
data = [];
labels = [];

% Number of samples for each class
n_0 = round(prior_0 * n_samples);
n_1 = round(prior_1 * n_samples);

% Generate class 0 samples
samples_01 = mvnrnd(m01, Cij, round(w(1) * n_0));
samples_02 = mvnrnd(m02, Cij, round(w(2) * n_0));
data = [data; samples_01; samples_02];
labels = [labels; zeros(size(samples_01, 1), 1); zeros(size(samples_02, 1), 1)];

% Generate class 1 samples
samples_11 = mvnrnd(m11, Cij, round(w(1) * n_1));
samples_12 = mvnrnd(m12, Cij, round(w(2) * n_1));
data = [data; samples_11; samples_12];
labels = [labels; ones(size(samples_11, 1), 1); ones(size(samples_12, 1), 1)];

% Compute likelihoods for class 0 and class 1
likelihood_0 = w(1) * mvnpdf(data, m01, Cij) + w(2) * mvnpdf(data, m02, Cij);
likelihood_1 = w(1) * mvnpdf(data, m11, Cij) + w(2) * mvnpdf(data, m12, Cij);

% Compute likelihood ratio
likelihood_ratio = likelihood_1 ./ likelihood_0;

% Initialize variables for ROC curve
gamma_values = logspace(-10, 10, 10000); % Gamma values for decision threshold
TPR_list = zeros(length(gamma_values), 1); % True Positive Rate list
FPR_list = zeros(length(gamma_values), 1); % False Positive Rate list
error_prob_list = zeros(length(gamma_values), 1); % Error probability list

% Compute TPR, FPR, and error probability for each gamma value
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    
    % Classification decision based on likelihood ratio and threshold gamma
    decision = likelihood_ratio > gamma;

    % Compute TP, FP, TN, FN
    TP = sum((decision == 1) & (labels == 1)); % True positives
    FP = sum((decision == 1) & (labels == 0)); % False positives
    TN = sum((decision == 0) & (labels == 0)); % True negatives
    FN = sum((decision == 0) & (labels == 1)); % False negatives

    % Compute TPR (True Positive Rate) and FPR (False Positive Rate)
    TPR = TP / (TP + FN); % Sensitivity or Recall
    FPR = FP / (FP + TN); % False Positive Rate
    FNR = FN / (FN + TP); % False Negative Rate

    % Compute error probability P(error; gamma)
    error_prob = FPR * prior_0 + FNR * prior_1; % Weighted by prior probabilities

    % Store TPR, FPR, and error probability
    TPR_list(i) = TPR;
    FPR_list(i) = FPR;
    error_prob_list(i) = error_prob;
end

% Find the minimum error probability and corresponding gamma value
[min_error_prob, min_idx] = min(error_prob_list); % Find minimum error probability
best_gamma = gamma_values(min_idx); % Corresponding gamma value

% Get the TPR and FPR for the minimum error probability
best_TPR = TPR_list(min_idx);
best_FPR = FPR_list(min_idx);

% Output the minimum error probability and best gamma
fprintf('Minimum Error Probability: %.4f at Gamma: %.4f\n', min_error_prob, best_gamma);
fprintf('Corresponding TPR: %.4f, FPR: %.4f\n', best_TPR, best_FPR);

% Plot ROC curve
figure;
plot(FPR_list, TPR_list, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve');
grid on;

% Plot the minimum error point on the ROC curve
hold on;
plot(best_FPR, best_TPR, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Use red circle to mark the minimum error point
legend('ROC Curve', 'Minimum Error Point', 'Location', 'Best');
hold off;

% Plot Error Probability curve
figure;
plot(gamma_values, error_prob_list, 'r-', 'LineWidth', 2);
xlabel('Gamma');
ylabel('Error Probability P(error)');
title('Error Probability Curve');
set(gca, 'XScale', 'log'); % Set x-axis to logarithmic scale
grid on;

% Mark the minimum error point on the error probability curve
hold on;
plot(best_gamma, min_error_prob, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
legend('Error Probability Curve', 'Minimum Error Point', 'Location', 'Best');
hold off;

% Define parameters
x_range = linspace(min(data(:,1))-1, max(data(:,1))+1, 200); % X-axis range
y_range = linspace(min(data(:,2))-1, max(data(:,2))+1, 200); % Y-axis range
[x_mesh, y_mesh] = meshgrid(x_range, y_range); % Create a grid

% Flatten the grid
grid_points = [x_mesh(:), y_mesh(:)];

% Compute likelihoods for class 0 and class 1 for grid points
likelihood_0_grid = w(1) * mvnpdf(grid_points, m01, Cij) + w(2) * mvnpdf(grid_points, m02, Cij);
likelihood_1_grid = w(1) * mvnpdf(grid_points, m11, Cij) + w(2) * mvnpdf(grid_points, m12, Cij);

% Compute likelihood ratio for grid points
likelihood_ratio_grid = likelihood_1_grid ./ likelihood_0_grid;

% Make classification decision for the grid points using the best_gamma
decision_grid = likelihood_ratio_grid > best_gamma;

% Reshape decision grid back to mesh size
decision_grid = reshape(decision_grid, size(x_mesh));

% Plot the decision boundary
figure;
gscatter(data(:,1), data(:,2), labels, 'rb', 'xo');
hold on;
contour(x_mesh, y_mesh, decision_grid, [0.5 0.5], 'k-', 'LineWidth', 2); % Decision boundary at likelihood ratio = best_gamma
xlabel('Feature 1');
ylabel('Feature 2');
title('Decision Boundary');
legend('Class 0', 'Class 1', 'Decision Boundary');
grid on;
hold off;
