# coding: utf-8

import numpy as np
from dataset.mnist import load_mnist
from simple_conv_net import SimpleConvNet
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001

train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30,
                                    'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

for i in range(iters_num):
    # print('progress ...' + str(i))
    # ミニバッチ取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)

    # パラメータ更新
    optimizer = AdaGrad(learning_rate)
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("=== epoch:" + str(i) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")


test_acc = network.accuracy(x_test, t_test)
print("=============== Final Test Accuracy ===============")
print("test acc:" + str(test_acc))
