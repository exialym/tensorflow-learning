# coding:utf-8
import tensorflow as tf

# 读取MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 实现softmax模型y = softmax(Wx + b) = normalize(exp(Wx + b))
# x 代表着所有图像数据，一幅图像有28 * 28 = 784个点，使用一个784维的向量代表一幅图
# 一幅图将是一个在784维空间中的向量，向量个数未知
x = tf.placeholder(tf.float32, [None, 784])
# W 代表784个像素点中每一个像素点的权重，对于每一种数字，每个像素点的权重是不同的
# 比如中心点，在数字0时，他的权重是负的，在数字8时，就可能是正的，所以每个像素点有10种权重
W = tf.Variable(tf.zeros([784, 10]))
# b是对于每一种数字的修正量，一共有10个
b = tf.Variable(tf.zeros([10]))
# 那么对于每一幅图像（x），我们把它和某个数字的权重矩阵相乘，再加上对应的修正，Wx + b。
# 就得到了这幅图像是这个数字的一个量化的证据，这个证据对于每幅图来说有10个。
# 我们把这个证据归一化，normalize(exp(Wx + b))，得到一个向量，这个向量有10维，代表这幅图是10个数字的概率
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_是我们的验证向量，是哪个数字，那个数字对应的位就是1
y_ = tf.placeholder(tf.float32, [None, 10])
# 我们使用交叉熵来作为模型的评判标准，计算所有y与y_的交叉熵，计算所有交叉熵的平均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 现在，因为 TensorFlow 拥有一张描述你各个计算单元的图
# 它可以自动地使用反向传播算法来有效地确定你的变量是如何影响你想要最小化的那个成本值的
# 然后，TensorFlow 会用你选择的优化算法来不断地修改变量以降低成本
# 我们要求 TensorFlow 用梯度下降算法以 0.05 的学习速率最小化交叉熵

# TensorFlow 在这里实际上所做的是：
# 它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法
# 然后，它返回给你的只是一个单一的操作
# 当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本，即交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化变量，会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 取1000次，每次随机取100个值，将图像和其代表的数字分别传给x和y_，训练你的W和b
# 目标是cross_entropy值最小
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 至此模型已经训练完了，W和b已经是当下最优的了
# 我们使用测试集测试一下
# tf.argmax返回向量中最大的值的下标
# tf.argmax(y, 1)就是我们的模型认为的，一幅图最可能是哪个数字
# 我们和真正的label比脚一下,有多少个y就会得到一个有多少个boolean元素的列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 使用cast将其转换成0，1。求下平均，就会得到准确率，比如[0,1]的准确率是50%
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 我们把测试图像传给x，计算出y。将labels传给y_。然后就可以测试我们的模型啦
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
