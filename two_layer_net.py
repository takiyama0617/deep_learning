import numpy as np
import sys, os
sys.path.append(os.pardir)
from gradient import numerical_gradient
from layer.affine_layer import Affine
from layer.relu_layer import Relu
from layer.softmax_with_loss_layer import SoftmaxWithLoss
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        Args: 
            input_size: 入力層のニューロン数
            hidden_size: 隠れ層のニューロン数
            output_size: 出力層のニューロン数
        """

        # ニューラルネットワークのパラメータを保持するディクショナリー変数
        self.params = {}
        # １層目の重み
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # １層目のバイアス
        self.params['b1'] = np.zeros(hidden_size)
        # ２層目の重み
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # ２層目のバイアス
        self.params['b2'] = np.zeros(output_size)

        # レイヤ生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        """
        ### 認識を行う
        Args:
            x: 画像データ
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        """
        ### 損失関数を求める
        Args:
            x: 画像データ
            t: 正解ラベル
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        """
        ### 認識精度を求める
        Args:
            x: 画像データ
            t: 正解ラベル
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """
        ### 重みパラメータに対する勾配を求める
        Args:
            x: 画像データ
            t: 正解ラベル
        """
        loss_W = lambda W: self.loss(x, t)

        # 勾配を保持するディクショナリー変数
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配を保持するディクショナリー変数
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


