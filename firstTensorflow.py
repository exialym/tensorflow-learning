# coding:utf-8
import tensorflow as tf

# 先设置好所有的节点，后生成图来计算

# 常数operation，这个op会被作为节点添加到默认图中
print("整个流程简单示例")
matrix1 = tf.constant([[3., 3.]])
# 这时打印出这个变量可以看到这是一个Tensor类型的对象
# 在TensorFlow中这个对象代表所有数据，可以把它看作是一个n维的数组或列表
print(matrix1)
# operation返回的对象代表着operation的输出，可以传递给其他的operation构造函数作为输入
matrix2 = tf.constant([[2.], [2.]])
# 矩阵乘法op，使用上面两个op的输出作为输入
product = tf.matmul(matrix1, matrix2)
print(product)
# 现在这个图有3个节点，两个常数op一个乘法op，真的要进行运算就要在一个session 中执行这个图
# 不传入任何参数的session会载入默认的图
sess = tf.Session()
# 将我们要执行的op的输出传进run，run会去计算这个op，并自动并行计算这个op的所有输入
# result = sess.run(matrix1)
# print(result)
result = sess.run(product)
print(result)
sess.close()

# 使用变量实现计数器
print("变量及Fetch示例")
# 计数器状态
state = tf.Variable(0, name="counter")
# 计数器步进
one = tf.constant(1)
# 计数器缓存
new_value = tf.add(state, one)
# 计数器更新
update = tf.assign(state, new_value)
# 变量初始化op
init_op = tf.global_variables_initializer()
sess = tf.Session()
# 变量要先初始化
sess.run(init_op)
print(sess.run(state))
for _ in range(3):
    # Fetch机制
    # 可以同时取回多个节点的值
    print(sess.run([state, update]))
# 通常会将一个统计模型中的参数表示为一组变量.
# 例如, 你可以将一个神经网络的 权重作为某个变量存储在一个 tensor 中.
# 在训练过程中, 通过重复运行训练图, 更新这个 tensor.

# Feeds机制
print("Feeds示例")
# 有时我们在定义op时并不知道我们的输入会是什么，这时我们可以先用占位符占上
input1 = tf.placeholder(tf.float32)
print(input1)
input2 = tf.placeholder(tf.float32)
# 并使用这些占位符用作op的输入
output = tf.mul(input1, input2)
sess = tf.Session()
print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))


# 可以使用InteractiveSession 代替Session类
# 使用 Tensor.eval()和 Operation.run() 方法代替 Session.run()
# 这样可以避免使用一个变量来持有会话.
print("交互式Session示例")
sess = tf.InteractiveSession()
a = tf.Variable([1.0, 2.0])
a.initializer.run()
b = tf.constant([1.0, 2.0])
sub = tf.sub(a, b)
print(sub.eval())
sess.close()


