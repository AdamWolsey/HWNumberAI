import numpy as np
import pickle

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(-1, 28 * 28)
        images = images / 255.0
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images_file = 'train-images-idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte'
test_images_file = 't10k-images-idx3-ubyte'
test_labels_file = 't10k-labels-idx1-ubyte'

train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    n_samples = targets.shape[0]
    log_p = - np.log(predictions[range(n_samples), targets])
    loss = np.sum(log_p) / n_samples
    return loss

def accuracy(predictions, targets):
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == targets)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    params = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * 0.01,
        'b2': np.zeros((1, output_size)),
    }
    return params

def forward_propagation(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def backward_propagation(X, y, params, cache):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]
    
    y_one_hot = np.zeros((m, 10))
    y_one_hot[range(m), y] = 1

    # Calculate gradients
    dZ2 = A2 - y_one_hot
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params

# Save the model parameters to a file
def save_model(params, filename='model_params.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {filename}")

# Load the model parameters from a file
def load_model(filename='model_params.pkl'):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model loaded from {filename}")
    return params

# Train the neural network
def train(X_train, y_train, X_test, y_test, hidden_size=128, learning_rate=0.1, epochs=100, batch_size=64, save=False, load=False, model_filename='model_params.pkl'):
    input_size = X_train.shape[1]
    output_size = 10
    
    # Load model if specified
    if load:
        params = load_model(model_filename)
    else:
        params = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch gradient descent
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            _, cache = forward_propagation(X_batch, params)

            grads = backward_propagation(X_batch, y_batch, params, cache)

            # Update parameters
            params = update_parameters(params, grads, learning_rate)

        # Evaluate after each epoch
        train_predictions, _ = forward_propagation(X_train, params)
        test_predictions, _ = forward_propagation(X_test, params)
        
        train_loss = cross_entropy_loss(train_predictions, y_train)
        test_loss = cross_entropy_loss(test_predictions, y_test)
        test_accuracy = accuracy(test_predictions, y_test)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Save the model if specified
    if save:
        save_model(params, model_filename)

    return params

# Training the neural network
params = train(train_images, train_labels, test_images, test_labels, hidden_size=128, learning_rate=0.1, epochs=20, batch_size=64, save=True, load=True, model_filename='mnist_model.pkl')
