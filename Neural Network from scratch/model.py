import numpy as np
import json

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Initialize weights and biases
        # The initialization for ReLU weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier/Glorot initialization for Softmax weights
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_deriv(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X):
        # X shape: (batch_size, input_size)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self, X, Y, learning_rate=0.1):
        # X shape: (batch_size, input_size)
        # Y shape: (batch_size, output_size) - One-hot encoded
        m = X.shape[0]
        
        # Output layer error
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1), np.max(output, axis=1)

    def save_weights(self, path="model_weights.json"):
        weights = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist()
        }
        with open(path, "w") as f:
            json.dump(weights, f)
            
    def load_weights(self, path="model_weights.json"):
        with open(path, "r") as f:
            weights = json.load(f)
            self.W1 = np.array(weights["W1"])
            self.b1 = np.array(weights["b1"])
            self.W2 = np.array(weights["W2"])
            self.b2 = np.array(weights["b2"])
