import numpy as np
import csv
import sys

def cross_entropy_loss(Y,Y_cap):
    final_sum = np.sum(np.multiply(Y, np.log(Y_cap)))
    m = Y.shape[1]
    L = -(1/m) * final_sum
    return L
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
def feed_forward(X, params):

    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = sigmoid(cache["Z2"])
    cache["Z3"] = np.matmul(params["W3"], cache["A2"]) + params["b3"]
    cache["A3"] = np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)

    return cache
def back_propagate(X, Y, params, cache):
    
    dZ3 = cache["A3"] - Y
    dW3 = (1./m_batch) * np.matmul(dZ3, cache["A2"].T)
    db3 = (1./m_batch) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.matmul(params["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    return grads
file1 = 'train_image.csv'
file2 = 'train_label.csv'
file3 = 'test_image.csv'
args = len(sys.argv)
if args>1:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
with open(file1, newline='') as f:
    reader = csv.reader(f)
    train_image = list(reader)
with open(file2, newline='') as f:
    reader = csv.reader(f)
    train_label = list(reader)
with open(file3, newline='') as f:
    reader = csv.reader(f)
    test_image = list(reader)
train_image = np.array(train_image)
m = train_image.shape[0]
train_image = train_image.astype(int)
train_image = (train_image*0.99/255)+0.01
train_image = train_image.T
test_image = np.array(test_image)
test_image = test_image.astype(int)
test_image = (test_image*0.99/255)+0.01
test_image = test_image.T
train_label = np.array(train_label)
examples = train_label.shape[0]
train_label = train_label.reshape(1, examples)
tl_new = np.eye(10)[train_label.astype('int32')]
tl_new = tl_new.T.reshape(10, examples)
input_nodes = train_image.shape[0]
hidden_layer_1 = 64
hidden_layer_2 = 32
#learning_rate = 0.99
learning_rate = 3
beta = .9
batch_size = 128
batches = -(-m // batch_size)
params = { "W1": np.random.randn(hidden_layer_1, input_nodes) * np.sqrt(1. / input_nodes),
           "b1": np.zeros((hidden_layer_1, 1)) * np.sqrt(1. / input_nodes),
           "W2": np.random.randn(hidden_layer_2, hidden_layer_1) * np.sqrt(1. / hidden_layer_1),
           "b2": np.zeros((hidden_layer_2, 1)) * np.sqrt(1. / hidden_layer_1), 
           "W3": np.random.randn(10, hidden_layer_2) * np.sqrt(1. / hidden_layer_2),
           "b3": np.zeros((10, 1)) * np.sqrt(1. / hidden_layer_2) 
           }
V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)
V_dW3 = np.zeros(params["W3"].shape)
V_db3 = np.zeros(params["b3"].shape)

X_train = train_image
Y_train = tl_new
for i in range(9):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])
        V_dW3 = (beta * V_dW3 + (1. - beta) * grads["dW3"])
        V_db3 = (beta * V_db3 + (1. - beta) * grads["db3"])


        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2
        params["W3"] = params["W3"] - learning_rate * V_dW3
        params["b3"] = params["b3"] - learning_rate * V_db3

    cache = feed_forward(X_train, params)
    train_cost = cross_entropy_loss(Y_train, cache["A3"])

cache = feed_forward(test_image, params)
result = np.argmax(cache["A3"], axis=0)
result = result.T
np.savetxt("test_predictions.csv", result, delimiter=',',fmt='%d')