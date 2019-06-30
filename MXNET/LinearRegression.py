"""
My attempt to implement simple Linear regression, I also tried to implement changing context to gpu but have some troubles ...
"""
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon import loss 
import mxnet as mx
import time


class LinearRegression():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.batch = None
        self.epochs = None
        self.W = None
        self.b = None
        self.learning_rate = None
        self.indices = None
    
    def fit(self, X,Y, epochs = 100, batch = 100, learning_rate = 0.01, every = 10):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.W = nd.random.normal(scale=0.01, shape=((X.shape[1],1)))
        self.b = nd.zeros(shape=(1,))
        self.W.attach_grad()
        self.b.attach_grad()
        self.batch = batch
        if self.batch > self.Y.shape[0]:
            self.batch = self.Y.shape[0]
        self.epochs = epochs
        self.indices = nd.arange(self.Y.shape[0])
        for _ in range(epochs):
            x, y = next(self.data_iter())
            with autograd.record():
                y_hat = nd.dot(x, self.W) + self.b
                loss = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
            loss.backward()
            #stochastic gradient descent
            for param in [self.W, self.b]:
                param[:] = param - self.learning_rate * param.grad / self.batch
            y_hat = nd.dot(x, self.W) + self.b
            loss = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
            if _ % every == 0:
                print("Epoch:",str(_), 'Loss {}'.format(loss.mean().asnumpy()))
    
    def predict(self, x):
        return nd.dot(x, self.W) + self.b
    
    def save(self):
        pass
    
    def data_iter(self):
        while True:
            nd.shuffle(self.indices, out = self.indices)
            inds = nd.array(self.indices[:self.batch])
            yield self.X.take(inds), self.Y.take(inds)

class LinearRegressionGluon():
    """
    using Gluon api
    """
    def __init__(self, ctx = mx.cpu(), learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.ctx = ctx
        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))
        self.net.initialize(init.Normal(sigma=0.01), ctx = self.ctx)
        self.loss = loss.L2Loss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.learning_rate})
    
    def fit(self, X, Y, epochs = 100, batch = 100, learning_rate = 0.01, every = 10):
        features = X.as_in_context(self.ctx)
        labels = Y.as_in_context(self.ctx)
        if learning_rate != self.learning_rate:
            self.learning_rate = learning_rate
            self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.learning_rate})
        #create iter class for sampling data as batches
        dataset = gluon.data.ArrayDataset(X,Y)
        data_iter =  gluon.data.DataLoader(dataset, batch, shuffle = True)
        for epoch in range(1, epochs + 1):
            for x, y in data_iter:
                x = x.as_in_context(self.ctx)
                y = y.as_in_context(self.ctx)
                with autograd.record():
                    l = self.loss(self.net(x), y)
                l.backward()
                self.trainer.step(batch)
            if epoch % every == 0:
                print('epoch %d, loss: %f' % (epoch, self.loss(self.net(features.as_in_context(self.ctx)), labels.as_in_context(self.ctx)).mean().asnumpy()))
        
    def predict(self, x):
        features = x.as_in_context(self.ctx)
        return self.net(features)
