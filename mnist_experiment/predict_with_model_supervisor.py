import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

# 定义卷积层
def conv_layer(input, weight_shape) :
    # 定义卷积核的权重w
    W_conv = weight_variable(weight_shape)
    tf.summary.histogram('W_conv', W_conv)
    # tf.summary.image('image_conv', W_conv, 10)
    # 定义bias. 一个卷积核一个bias
    bias_shape = weight_shape[3]
    b_conv = bias_variable([bias_shape])
    tf.summary.histogram('b_conv', b_conv)

    # conv2d
    h_conv = tf.nn.relu( tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    # max_pool_2x2
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_pool


if __name__ == '__main__' :
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    # 构建阶段
    # 定义输入数据：x和y
    with tf.name_scope('input') :
        x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        y = tf.placeholder(tf.float32, [None, 10], name='input_y')

    # 规范化输入图片的shape: 28*28*1
    x_image = tf.reshape(x, [-1,28,28,1])
    tf.summary.image('image_input', x_image, 10)

    ########## 定义模型 ##########################################################
    with tf.name_scope('model') :
        ### 第一层卷积层：32个 5*5 的卷积核
        with tf.name_scope('layer1_conv'):
            layer1_output = conv_layer(x_image, [5, 5, 1, 32])
            # layer1_output 的维度是[-1, 14, 14, 32]

        ### 第二层卷积层：64个 5*5 的卷积核
        with tf.name_scope('layer2_conv'):
            layer2_output = conv_layer(layer1_output, [5, 5, 32, 64])
            # layer2_output 的维度应该是[-1, 7, 7, 64]

        ####### 第三层全连接层：1024个节点
        with tf.name_scope('layer3_fc'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(layer2_output, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # h_fc1的维度是[-1, 1024]

        # 第四层dropout层
        with tf.name_scope('layer4_dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 第五层softmax层：全连接层+softmax变换，输出节点为10个
        with tf.name_scope('layer5_softmax'):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # 定义loss
        with tf.name_scope('compute_loss'):
            # 定义交叉熵
            cross_entropy_op = -tf.reduce_sum(y * tf.log(y_pred), name='loss_op')
            # 监控交叉熵
            loss_scalar = tf.summary.scalar('loss', cross_entropy_op)
            # 定义优化器和训练器
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_op)

    # 定义accuracy
    with tf.name_scope('compute_accuracy'):
        # 准确率的计算方式
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
        # 监控acc
        accuracy_scalar = tf.summary.scalar('accuracy', accuracy_op)


    ########## 启动会话 #################################################***#########
    # log保存路径
    logs_path = 'logs'

    # 定义Supervisor
    sv = tf.train.Supervisor(logdir=logs_path, init_op=tf.global_variables_initializer(), summary_op=None )
    # 启动会话
    with sv.managed_session() as sess :

        ####### 对测试集进行测试 ##################################
        import tensorflow.examples.tutorials.mnist.input_data as input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        # 整体一次性测试
        # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # print("testSet accuracy %g" % test_accuracy )
        
        # 分段测试. 避免内存不够大
        # test_accuracys = []
        # BATCH_SIZE = 64
        # for i in range(0, int(mnist.test.images.shape[0]), BATCH_SIZE):
        #     test_images = mnist.test.images[i:i+BATCH_SIZE]
        #     test_labels = mnist.test.labels[i:i+BATCH_SIZE]
        #     test_accuracy = sess.run(accuracy_op, feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})
        #     test_accuracys.append(test_accuracy)
        # print("testSet accuracy :%g" % ( sum(test_accuracys)/len(test_accuracys) ))
        
        # 单张图片测试
        need = 233
        test_image = mnist.test.images[need]
        test_label = mnist.test.labels[need]
        print(test_image)
        print(test_label)

        test_image = test_image.reshape(-1, 28 * 28)
        test_label = test_label.reshape(-1, 10)
        y_pred = sess.run(y_pred, feed_dict={x: test_image, keep_prob: 1.0})
        # acc = sess.run(accuracy_op, feed_dict={x: test_image, y: test_label, keep_prob: 1.0})

        print('test_label:{}, y_pred:{}'.format(test_label.argmax(), y_pred.argmax()))
        # print(acc)
        