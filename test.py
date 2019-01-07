import tensorflow as tf

# a = tf.Variable(tf.random_normal([2,3,4,5]))
# b,c = tf.nn.moments(a,axes=[0,1],keep_dims=False)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run([b,c]))


# scale = tf.get_variable("scale", [2], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
# a = scale.get_shape()  # a.get_shape()中a的数据类型只能是tensor，且返回的是一个元组
# b = tf.shape(scale)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(b))
input = tf.Variable(tf.random_uniform((30,256,256,3)))
a = tf.contrib.slim.conv2d(input, 64, 3, 2, padding='SAME', activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=None)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(tf.shape(input)))
    sess.run(a)
    print(sess.run(tf.shape(a)))

