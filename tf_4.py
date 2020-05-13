# -*- coding: utf-8 -*-
# 导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 32  # (每次计算个数)
seed = 23455

# 基于seed产生随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵，表示32组 体积和重量 作为输入数据集
X = rng.rand(32, 2)
# 从X中取出1行，判断如果和小于1，给Y赋值1（表示合格），如果和大于等于1，给Y赋值0（表示不合格）
# 作为输入数据集的标签（正确答案）
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X: ", X)
print("Y: ", Y)

# 定义输入和参数,输出，前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))   # 输入多个实例（样本属性）
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 输入多个实例（样本标签）
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)  # 计算出的标签

# 定义损失函数和反向传播
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 生成会话训练模型
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化所有变量
    sess.run(init_op)

    # 输出目前训练前的参数值
    print("w1: ", sess.run(w1))
    print("w2: ", sess.run(w2))
    print("\n")

    # 训练模型
    STEPS = 10000    # 训练次数 3000轮
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print ("After %d training steps, loss on all data is %g" % (i, total_loss))
    # 输出训练后的参数值
    print("\n")
    print("w1: ", sess.run(w1))
    print("w2: ", sess.run(w2))
    print("\n")
