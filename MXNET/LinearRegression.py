"""
My attempt to implement simple Linear regression, I also tried to implement changing context to gpu but have some troubles ...
"""

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
