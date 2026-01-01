import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import model
import time

def load_data():
    print("Loading MNIST dataset... this might take a minute.")
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Normalize inputs to 0-1
    X = X / 255.0
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False, categories='auto')
    y = enc.fit_transform(y.reshape(-1, 1))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    nn = model.NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    
    epochs = 50
    batch_size = 64
    learning_rate = 0.1
    decay_rate = 0.95
    
    n_samples = X_train.shape[0]
    
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Learning Rate Decay every 10 epochs
        if (epoch + 1) % 10 == 0:
            learning_rate *= decay_rate
            print(f"Decaying learning rate to {learning_rate:.5f}")
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        total_loss = 0
        
        # Mini-batch gradient descent
        for i in range(0, n_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            output = nn.forward(X_batch)
            nn.backward(X_batch, y_batch, learning_rate)
            
            # Simple Cross Entropy Loss for monitoring
            epsilon = 1e-15
            output_clipped = np.clip(output, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_batch * np.log(output_clipped), axis=1))
            total_loss += loss
            
        avg_loss = total_loss / (n_samples / batch_size)
        
        # Calculate accuracy on test set
        test_predictions, _ = nn.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(test_predictions == y_test_labels)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy*100:.2f}% - Time: {time.time() - start_time:.2f}s")
        
    print("\n" + "="*30)
    print("       FINAL REPORT       ")
    print("="*30)
    print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
    print("="*30 + "\n")
    
    print("Training complete!")
    nn.save_weights("model_weights.json")
    print("Weights saved to model_weights.json")

if __name__ == "__main__":
    train()
