import time
import tensorflow as tf

def weight_variable(shape):
  # 由于使用ReLU激活函数，所以我们通常将这些参数初始化为很小的正值。
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weights')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='bias')



# 定义卷积层
def conv_layer(input, weight_shape) :
    # 定义卷积核的权重w     # 由于使用ReLU激活函数，所以我们通常将这些参数初始化为很小的正值。
    W_conv = weight_variable(weight_shape)
    tf.summary.histogram('W_conv', W_conv)
    # tf.summary.image('image_conv', W_conv, 10)
    # 一个卷积核一个bias
    bias_shape = weight_shape[3]
    b_conv = bias_variable([bias_shape])
    tf.summary.histogram('b_conv', b_conv)

    # conv2d
    h_conv1 = tf.nn.relu( tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    # max_pool_2x2
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_pool1



if __name__ == '__main__' :
    # 构建阶段
    # mnist.train.images 等等 都是ndarray类型
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 输入数据：x和y
    x = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='input_y')


    # 规范化输入图片的shape: 28*28*1
    x_image = tf.reshape(x, [-1,28,28,1])
    tf.summary.image('image_input', x_image, 10)

    ####### 第一层 卷积+池化
    with tf.name_scope('layer1_conv'):
        layer1_output = conv_layer(x_image, [5, 5, 1, 32])
        # layer1_output 的维度是[-1, 14, 14, 32]

    ####### 第二层 卷积+池化
    with tf.name_scope('layer2_conv'):
        layer2_output = conv_layer(layer1_output, [5, 5, 32, 64])
        # layer2_output 的维度应该是[-1, 7, 7, 64]

    ####### 第三层全连接层
    with tf.name_scope('layer3_fc') :
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(layer2_output, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # h_fc1的维度应该是[-1, 1024]

    # 第四层dropout层
    with tf.name_scope('layer4_dropout') :
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(keep_prob)

    # 第五层fc层+softmax变换
    with tf.name_scope('layer5_softmax') :
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        print(y_output)

    # 定义loss
    with tf.name_scope('loss') :
        # 定义交叉熵
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_output))
        # 监控交叉熵
        loss_scalar = tf.summary.scalar('loss', cross_entropy)
        # 定义训练器
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    ###### 执行阶段
    # 定义准确率的计算方式
    with tf.name_scope('accuracy') :
        # 定义acc
        correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy)
        # 监控acc
        accuracy_scalar = tf.summary.scalar('accuracy', accuracy)


    # # 定义保存器
    saver = tf.train.Saver(max_to_keep=3)
    # # 模型保存路径
    save_path = 'ckpt/'
    # # 当前最高精确度
    # saver_max_acc = 0

    # tenrsorboard可视化
    summary_merged = tf.summary.merge_all()
    # log保存路径
    logs_path = 'logs/'

    # 启动会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess :
        train_writer = tf.summary.FileWriter(logs_path+'train', sess.graph)
        validation_writer = tf.summary.FileWriter(logs_path+'validation')
        sess.run(tf.global_variables_initializer())

        # 超参数
        EPOCH = 2
        SAMPLE_NUM = mnist.train.images.shape[0]
        # ITERATION = 500 +1
        # BATCH_SIZE = 128
        BATCH_SIZE = 64
        ITERATION_SHOW = 100
        start_time = time.time()
        for step in range(EPOCH) :
            for _ in range( int(SAMPLE_NUM/BATCH_SIZE)):
                batch = mnist.train.next_batch(BATCH_SIZE)
                # 训练阶段dropout设置为0.5,预测阶段设置为1.0
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            # if step%ITERATION_SHOW == 0:
            end_time = time.time()

            # 计算当前训练集的准确率
            summary, accuracy_currut_train = sess.run([summary_merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # 写入summary
            train_writer.add_summary(summary, step)

            # 计算当前验证集的准确率
            vat_batch = mnist.validation.next_batch(BATCH_SIZE)
            (sum_accuracy_validation,
             sum_loss_validation,
             accuracy_currut_validation) = sess.run([accuracy_scalar, loss_scalar, accuracy], feed_dict={x:mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            # 写入summary
            validation_writer.add_summary(sum_accuracy_validation, step)
            validation_writer.add_summary(sum_loss_validation, step)


            # 输出当前准确率
            # print("step %d, training accuracy :%.4g, used time: %.2gs" % (step, train_accuracy_v, end_time - start_time))
            timediff = end_time-start_time
            print("step %d, training accuracy :%.4g, validation accuracy :%.4g, used time: %.2gs"%(step, accuracy_currut_train, accuracy_currut_validation, timediff))

            start_time = time.time()

            # 选择准确度最高的模型 进行保存
            # if train_accuracy > saver_max_acc:
            saver.save(sess, save_path+'mnist_model', global_step=step)
              # saver_max_acc = train_accuracy

        # 关闭summary
        train_writer.close()
        validation_writer.close()

        # 对测试集进行测试
        # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        # print("test dataSets accuracy %.4g" % test_accuracy )
        test_accuracys = []
        for i in range(0, int(mnist.test.images.shape[0]), BATCH_SIZE):
            test_images = mnist.test.images[i:i+BATCH_SIZE]
            test_labels = mnist.test.labels[i:i+BATCH_SIZE]
            test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
            test_accuracys.append(test_accuracy)
        print("test accuracy %g" % ( sum(test_accuracys)/len(test_accuracys) ))

