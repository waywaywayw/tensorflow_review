import time
import tensorflow as tf


# 设置权重参数
def weight_variable(shape):
  # 由于使用ReLU激活函数，所以我们通常将这些参数初始化为很小的正值。
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积层和池化层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# 构建模型阶段
# mnist.test.images 等等 都是ndarray类型
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入数据：x和y
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None,10])


####### 第一层卷积
# 定义卷积核的权重w
W_conv1 = weight_variable([5, 5, 1, 32])
# 一个卷积核一个bias
b_conv1 = bias_variable([32])

# 规范化输入图片的shape: 28*28*1
x_image = tf.reshape(x, [-1,28,28,1])

# 定义第一层卷积层的具体实现
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_conv1的维度应该是[-1, 28, 28, 32]
h_pool1 = max_pool_2x2(h_conv1)
# h_pool1的维度应该是[-1, 14, 14, 32]


####### 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_pool2的维度应该是[-1, 7, 7, 64]


####### 第三层全连接层(full connection) 或者叫 密集连接层( denses )
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1的维度应该是[-1, 1024]


# 第四层dropout层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 第五层fc层+softmax变换
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 定义交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 定义训练器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 定义准确率的计算方式
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


### 执行会话阶段
# 设置GPU需求自动增长，避免申请过多GPU内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 开启会话
with tf.Session(config=config) as sess :
    sess.run(tf.global_variables_initializer())

    # 加入start_time 和 end_time 是为了方便计时
    start_time = time.time()
    # iteration
    for i in range(10000):
      # set batch_size
      batch = mnist.train.next_batch(32)
      if i%500 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})

        end_time = time.time()
        print("step %d, training accuracy %.4g, used time: %.2fs"%(i, train_accuracy, end_time-start_time))
        start_time = time.time()

      # run
      # 训练阶段dropout设置为0.5,预测阶段设置为1.0
      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 对测试集进行测试
    # 整体测试
    # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    # print("test accuracy %g" % test_accuracy )
    # 分段测试。防止内存不够..
    # 分段测试
    test_accuracys = []
    for i in range(0, int(mnist.test.images.shape[0]), 32):
        test_images = mnist.test.images[i:i + 32]
        test_labels = mnist.test.labels[i:i + 32]
        test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
        test_accuracys.append(test_accuracy)
    print("test accuracy :%g" % (sum(test_accuracys) / len(test_accuracys)))
