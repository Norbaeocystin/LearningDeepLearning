import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, rnn
from mxnet.contrib import text
import time
import os
import logging


logging.basicConfig(level=logging.DEBUG,  format = '%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger()

class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, ctx, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.ctx = ctx
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2, activation="sigmoid")
        self.ctx = ctx

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # Since the input (embeddings) is the only argument passed into
        # rnn.LSTM, it only returns the hidden states of the last hidden layer
        # at different time step (outputs). The shape of outputs is
        # (number of words, batch size, 2 * number of hidden units).
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * number of hidden units)
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.softmax(self.decoder(encoding))
        return outs
    
    def softmax(self, X):
        X_exp = nd.exp(X)
        partition = X_exp.sum(axis=1, keepdims=True)
        return X_exp / partition
    
class SentimentRNN(object):
    
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, ctx, embeds, clip_gradient = 5):
        self.ctx = ctx
        self.net = BiRNN(vocab_size, embed_size, num_hiddens, num_layers, self.ctx)
        self.net.initialize(init.Xavier(), ctx= self.ctx)
        self.embeds = embeds
        self.net.embedding.weight.set_data(self.embeds)
        self.net.embedding.weight.reset_ctx(self.ctx)
        self.net.embedding.collect_params().setattr('grad_req', 'null')
        self.clip_gradient = clip_gradient
        
    def accuracy(self, y_hat, y):
        return ((y_hat.argmax(axis = 1).reshape(-1,1) == y).sum()/y.shape[0]).asscalar()
        
    def fit(self, X,Y, epochs = 101, batch = 100, learning_rate = 0.1, every = 10):
        self.X = X
        self.Y = Y
        self.indices = nd.arange(Y.shape[0])
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': learning_rate, 
                                                                         'clip_gradient':self.clip_gradient},
                                                                         compression_params={'type':'2bit', 'threshold':0.5})
        #self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        def cross_entropy(y_hat, y):
            return - nd.pick(y_hat, y).log()
        self.batch = batch
        if self.batch > self.Y.shape[0]:
            self.batch = self.Y.shape[0]
        self.epochs = epochs
        dataset = gluon.data.ArrayDataset(X,Y)
        data_iter =  gluon.data.DataLoader(dataset, batch, shuffle = True)
        for epoch in range(1, epochs + 1):
            if isinstance(self.ctx, list):
                for x, y in data_iter:
                    features = gluon.utils.split_and_load(x, self.ctx)
                    labels = gluon.utils.split_and_load(y, self.ctx)
                    with autograd.record():
                        losses = [cross_entropy(self.net(X), Y) for X, Y in zip(features, labels)]
                        for l in losses:
                            l.backward()
                    self.trainer.step(batch, ignore_stale_grad=True)
                    #nd.waitall()
                if epoch % every == 0:
                    loss = [cross_entropy(self.net(X), Y) for X, Y in zip(features, labels)]
                    loss_modified = [item.mean().asscalar() for item in loss ]
                    acc = sum([self.accuracy(self.net(X), Y) for X, Y in zip(features, labels)])/len(self.ctx)
                    logger.debug('epoch {} loss: {}, acc {}'.format( epoch, loss_modified, acc))
                    del loss
                    del loss_modified
                    del acc
                del features
                del labels
            else:
                for x, y in data_iter:
                    x.as_in_context(self.ctx)
                    y.as_in_context(self.ctx)
                    with autograd.record():
                        l = cross_entropy(self.net(x), y)
                    l.backward()
                    self.trainer.step(batch, ignore_stale_grad=True)
                if epoch % every == 0:
                    loss = cross_entropy(self.net(x), y).mean().asnumpy()
                    acc = self.accuracy(self.net(x), y)
                    logger.debug('epoch {} loss: {}, acc {}'.format(epoch, loss, acc))
                    del loss
                    del acc
                del x
                del y

    def data_iter(self):
        while True:
            nd.shuffle(self.indices, out = self.indices)
            inds = nd.array(self.indices[:self.batch], ctx = self.ctx)
            yield self.X.take(inds), self.Y.take(inds)      
        
    def predict(self,X):
        label = self.net(X)
        return label
