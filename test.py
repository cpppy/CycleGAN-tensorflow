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


# input = tf.Variable(tf.random_uniform((30,256,256,3)))
# a = tf.contrib.slim.conv2d(input, 64, 3, 2, padding='SAME', activation_fn=None,
#                             weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                             biases_initializer=None)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(tf.shape(input)))
#     sess.run(a)
#     print(sess.run(tf.shape(a)))

# class Args_l(object):
#     __slots__ = ('dataset_dir', 'epoch','epoch_step','batch_size','train_size','load_size','fine_size','ngf',
#                  'ndf','input_nc', 'output_nc','lr','beta1','which_direction','phase','save_freq','print_freq',
#                  'continue_train','checkpoint_dir','sample_dir','test_dir','L1_lambda','use_resnet','use_lsgan',
#                  'max_size')
#     def __init__(self):
#         self.dataset_dir = 'horse2zebra'
#
#     def

# class Student():
#     # def __init__(self):
#     #     pass
#     @property
#     def score(self):
#         return self._score
#     @score.setter
#     def score(self,value):
#         if not isinstance(value,int):
#             raise ValueError('score must be an integer!')
#         if value<0 or value>100 :
#             raise  ValueError('value must between 0~100')
#         self._score = value
#
# a = Student()
# a.score = 90
# print(a.score)
# a.score = 110
# print(a.score)

class Screen(object):
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self,width):
        self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def resolution(self):
        return self._width*self._height
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')





