# -*- coding: utf-8 -*-
# 损失函数设为loss=（w+1)^2，令w初值为10， 反向传播求最优的w，使得loss最小时的w。
import tensorflow as tf
LEARNING_RATE_BASE = 0.1    # 最初的学习速率
LEARNING_RATE_DECAY = 0.99  # 学习速率的衰减速度
LEARNING_RATE_STEP = 1      # 喂入多少轮BATCH_SIZE之后，更新一次学习速率，一般设为总样本数/BATCH_SIZE

# 运行了几轮BATCH_SIZE的计数器，初始值为0， 设为不被训练
global_step = tf.Variable(0, trainable=False)
# 定义指数下降学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
# 定义优化参数，初始值为 10
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数loss
loss = tf.square(w+1)
# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化所有变量
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print ("After %s steps: global_step is %f, w is %f, learning_rate is %f, loss is %f" %
               (i, global_step_val, w_val, learning_rate_val, loss_val))