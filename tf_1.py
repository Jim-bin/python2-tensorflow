# -*- coding: utf-8 -*-
import tensorflow as tf
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0, 4.0]])
result = a + b

print(result)

c = tf.constant([[1.0, 2.0]])
d = tf.constant([[3.0], [4.0]])

y = tf.matmul(c, d)
print(y)

e = tf.constant([[3.0, 4.0]])
f = tf.constant([[5.0, 6.0]])

g = tf.add(e, f)
print(g)

