import tensorflow as tf

# 构建模型阶段
# 载入数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入数据：x
x = tf.placeholder(tf.float32, [None, 784])
# 输入数据：y
y_ = tf.placeholder("float", [None,10])

# 权重参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 定义模型。得到预测的y
y = tf.nn.softmax(tf.matmul(x,W) + b)


# 定义损失函数：交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 定义优化器和训练器
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 定义准确率的计算方式
# 取预测值和真实值 概率最大的标签
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 执行会话阶段
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# i 表示 iteration
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  if i%1000==0 :
    print("%d iteration, accuarcy:%.4f " % (i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) )