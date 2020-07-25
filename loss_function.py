import numpy as np
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# データを有効活用して、規則性を見出す
#   その一つの方法として、画像から特徴量を抽出して、そのパターンを機械学習で学習させる
#   ここで言う特徴量とは、
#     入力データから本質的なデータを的確に抽出できるように設計された変換器を指す
# ニューラルネットワークは、画像に含まれる特徴量も機械が学習する

# 損失関数
#   ２乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 損失関数
#   交差エントロピー誤差
def cross_entropy_error(y, t):
  if y.ndim = 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  # -infを防ぐ為、deltaを足す
  return -np.sum(t * np.log(y + delta)) / batch_size

# なぜ損失関数を設定するのか？？
#   微分が関係あるみたい
#   認識精度を指標にするとパラメーターの微分がほとんどの場所で０になってしまうから

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]