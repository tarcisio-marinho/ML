import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data(".", one_hot=True)

# hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# de 0 até 10
n_classes = 10

# de 100 em 100 features, alimentar a rede
batch_size = 100

# altura x largura = 28 x 28
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # (input_data * weights) + biases

    # tensor -> array com os pesos e bias
    hidden_1_layer = {"weights" : tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      "biases" : tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {"weights" : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases" : tf.Variable(tf.random_normal(n_nodes_hl2))}
    
    hidden_3_layer = {"weights" : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases" : tf.Variable(tf.random_normal(n_nodes_hl3))}
    
    output_layer = {"weights" : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      "biases" : tf.Variable(tf.random_normal([n_classes]))}
    


