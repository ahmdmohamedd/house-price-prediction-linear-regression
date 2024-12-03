%% Step 1: Load and Preprocess the Data
data = readtable('house_prices.csv'); % Load dataset

% Exclude non-predictive columns
data.Home = []; % Remove the 'Home' column

% Separate target variable
target = data.Price; % 'Price' is the target
data.Price = []; % Remove target from feature set

% Handle categorical variables (Brick, Neighborhood)
% Convert categorical columns to dummy variables
data.Brick = categorical(data.Brick); % Convert Brick to categorical
data.Neighborhood = categorical(data.Neighborhood); % Convert Neighborhood to categorical

% Encode categorical variables
encodedCategorical = [dummyvar(data.Brick), dummyvar(data.Neighborhood)];

% Extract and normalize numeric features
numericFeatures = data(:, ~ismember(data.Properties.VariableNames, {'Brick', 'Neighborhood'}));
numericFeatures = (table2array(numericFeatures) - mean(table2array(numericFeatures))) ./ ...
                  std(table2array(numericFeatures));

% Combine processed features
processedFeatures = [numericFeatures, encodedCategorical];

% Convert to numeric array
X = processedFeatures;
y = target;

%% Step 2: Split Data into Training and Test Sets
splitRatio = 0.8;
splitIdx = floor(splitRatio * size(X, 1));

X_train = X(1:splitIdx, :);
y_train = y(1:splitIdx);

X_test = X(splitIdx+1:end, :);
y_test = y(splitIdx+1:end);

% Add bias term (column of ones) to X
X_train = [ones(size(X_train, 1), 1), X_train];
X_test = [ones(size(X_test, 1), 1), X_test];

%% Step 3: Initialize Parameters
[m, n] = size(X_train);
theta = rand(n, 1); % Random initialization
alpha = 0.01; % Learning rate
num_iters = 1000; % Number of iterations
lambda = 1; % Regularization parameter

%% Step 4: Perform Gradient Descent with Regularization
cost_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Predictions
    predictions = X_train * theta;
    
    % Cost Function (MSE + Regularization)
    errors = predictions - y_train;
    cost = (1 / (2 * m)) * sum(errors .^ 2) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2); % Exclude bias term from regularization
    cost_history(iter) = cost;
    
    % Gradient Descent with Regularization
    gradients = (1 / m) * (X_train' * errors) + (lambda / m) * [0; theta(2:end)]; % Exclude bias term from regularization
    theta = theta - alpha * gradients;
end

%% Step 5: Evaluate the Model
y_pred_test = X_test * theta;

% Metrics
mae = mean(abs(y_pred_test - y_test));
rmse = sqrt(mean((y_pred_test - y_test).^2));

fprintf('Mean Absolute Error: %.2f\n', mae);
fprintf('Root Mean Squared Error: %.2f\n', rmse);

%% Step 6: Visualizations
% Plot Cost Function Convergence
figure;
plot(1:num_iters, cost_history, '-b', 'LineWidth', 2);
xlabel('Number of Iterations');
ylabel('Cost (MSE + Regularization)');
title('Cost Function Convergence');

% Predicted vs Actual Prices
figure;
scatter(y_test, y_pred_test, 'r');
hold on;
plot(y_test, y_test, '-b'); % Ideal line
xlabel('Actual Prices');
ylabel('Predicted Prices');
title('Actual vs Predicted Prices');
legend('Predictions', 'Ideal');
grid on;
