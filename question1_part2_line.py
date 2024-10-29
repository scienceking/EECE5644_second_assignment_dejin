#% This code is for question1 part 2, the linear part————————dejin wang

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Device selection: Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate 2000 training samples
X_train, y_train = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Generate 10,000 validation samples
X_val, y_val = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Convert training and validation datasets to PyTorch tensors and move to GPU (if available)
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)  # Reshape y_train into a column vector

X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)  # Reshape y_val into a column vector

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Input dimension is the feature dimension, output is 1

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Use the sigmoid function to compute probabilities

# Create the model instance and move it to GPU (if available)
input_dim = X_train.shape[1]  # Input feature dimension
model = LogisticRegressionModel(input_dim).to(device)

# Define the negative log-likelihood loss (calculated using binary cross-entropy) and move to GPU
criterion = nn.BCELoss().to(device)  # Binary cross-entropy loss is equivalent to negative log-likelihood loss
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Use stochastic gradient descent

# Train the model
num_epochs = 2000
loss_history = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute loss
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())  # Record the loss

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Make predictions on the validation set and calculate accuracy
with torch.no_grad():  # No need to compute gradients during validation
    y_val_pred = model(X_val_tensor)
    y_val_pred_class = (y_val_pred >= 0.5).float()  # Convert output to 0 or 1
    accuracy = (y_val_pred_class == y_val_tensor).float().mean()  # Compute accuracy, convert to float type
    print(f'Validation Accuracy: {accuracy.item() * 100:.2f}%')

# Visualize the loss curve
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Function to visualize decision boundaries
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid).to(device)

    with torch.no_grad():
        probs = model(grid_tensor).reshape(xx.shape)

    plt.contourf(xx, yy, probs.cpu(), levels=[0, 0.5, 1], alpha=0.6)  # Plot on CPU
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()

# Visualize the decision boundary on the training set
plot_decision_boundary(X_train, y_train, model, title="Decision Boundary on Training Set")

# Visualize the decision boundary on the validation set
plot_decision_boundary(X_val, y_val, model, title="Decision Boundary on Validation Set")
