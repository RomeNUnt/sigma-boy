import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(42)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)

bias_hidden = np.random.rand(1, hidden_neurons)
bias_output = np.random.rand(1, output_neurons)

def forward_propagation(X):
    global hidden_layer_activation, hidden_layer_output, final_output
    
    # Hidden layer computation
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    # Output layer computation
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = sigmoid(output_layer_activation)
    
    return final_output

def backpropagation(X, y, learning_rate=0.1):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    
    # Compute error
    output_error = y - final_output  
    output_delta = output_error * sigmoid_derivative(final_output)
    
    # Compute error for hidden layer
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Expected output

# Train for 10,000 epochs
epochs = 10000
for epoch in range(epochs):
    forward_propagation(X)
    backpropagation(X, y)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(y - final_output))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

output = forward_propagation(X)
print("Final Predictions:")
print(output)