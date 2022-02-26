# -*- coding:utf-8 -*-
import tensorflow as tf

hello = tf.constant('Hello World!')
s = tf.Session()
print(s.run(hello))