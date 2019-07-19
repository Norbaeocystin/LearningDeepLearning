import time
from mxnet import nd, gluon, autograd

def relu(X):
    return nd.maximum(X,0)

def prelu(x):
    return (nd.greater(x,0) * x) + (0.1 * (nd.lesser_equal(x,0) * x))


class MultilayerPerceptron(object):
    
    def __init__(self, neurons, activation_function = prelu, stdev_init = 0.01):
        self.activation_function = activation_function
        #if you want 2 layers neurons could be [64,32,32] 
        #if you want three layers neurons should be something like [128,128,64,64,32]
        # first element is input dimension and last element is output dimension
        self.neurons = neurons
        assert hasattr(self.neurons, '__iter__')
        self.layers = []
        for idx in range(len(self.neurons) - 1):
            w = nd.random.normal(scale=stdev_init, shape=(self.neurons[idx], self.neurons[idx + 1]))
            b = nd.random.normal(scale=stdev_init, shape=(self.neurons[idx + 1]))
            w.attach_grad()
            b.attach_grad()
            self.layers.append([w,b])
            
    def data_iter(self):
        while True:
            nd.shuffle(self.indices, out = self.indices)
            inds = nd.array(self.indices[:self.batch])
            yield self.X.take(inds), self.Y.take(inds)
            
    def fit(self, X,Y, epochs = 100, batch = 100, learning_rate = 0.01, every = 10):
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.batch = batch
        if self.batch > self.Y.shape[0]:
            self.batch = self.Y.shape[0]
        self.epochs = epochs
        self.indices = nd.arange(self.Y.shape[0])
        for _ in range(epochs):
            x, y = next(self.data_iter())
            with autograd.record():
                y_hat = self.predict(x)
                loss = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
            loss.backward()
            #stochastic gradient descent
            for (weight, bias) in self.layers:
                weight[:] = weight - self.learning_rate * weight.grad / self.batch
                bias[:] = bias - self.learning_rate * bias.grad / self.batch
            y_hat = self.predict(X)
            loss = (y_hat - Y.reshape(y_hat.shape)) ** 2 / 2
            #correct_prediction = (Y.reshape(y_hat.shape) - y_hat) ** 2
            #unexplained_error = nd.sum(correct_prediction)
            #total_error = nd.sum((Y - Y.mean()) ** 2)
            #R_squared = 1 -  (unexplained_error/total_error)
            if _ % every == 0:
                print(time.ctime(), "Epoch:",str(_), 'Loss {}'.format(loss.mean().asnumpy()))
        
        
    def predict(self, X):
        X = X.reshape((-1, self.neurons[0]))
        for (w,b) in self.layers:
            X = relu(nd.dot(X, w) +b)
        return X
