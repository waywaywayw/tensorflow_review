import os
import tensorflow as tf

# log保存路径
logs_path = 'logs_test'
train_path = os.path.join(logs_path, 'train')

# 加载图结构
ckpt = tf.train.get_checkpoint_state(train_path)
# print(ckpt.model_checkpoint_path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')

# 定义FileWriter。为了显示出测试集的图像
test_writer = tf.summary.FileWriter(os.path.join(logs_path, 'test'))
# 启动会话
with tf.Session() as sess:
    # 从最新的检查点 恢复会话
    saver.restore(sess, tf.train.latest_checkpoint(train_path))

    # 获取默认图, 以恢复需要的op
    graph = tf.get_default_graph()
    # 恢复输入节点
    x = graph.get_tensor_by_name("input/input_x:0")
    image_summary_op = graph.get_tensor_by_name('image_input:0')
    y = graph.get_tensor_by_name("input/input_y:0")
    # 恢复dropout节点
    keep_prob= graph.get_tensor_by_name('model/layer4_dropout/keep_prob:0')
    # 恢复输出节点
    y_pred = graph.get_tensor_by_name('model/layer5_softmax/Softmax:0')
    accuracy_op = graph.get_tensor_by_name('compute_accuracy/accuracy_op:0')


    ####### 对测试集进行测试 #####################################
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("data/fashion", one_hot=True)

    # 整体一次性测试
    # test_accuracy = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
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

    test_image = test_image.reshape([-1, 28*28])
    test_label = test_label.reshape([-1, 10])
    image_summary, y_pred = sess.run([image_summary_op, y_pred], feed_dict={x: test_image, keep_prob: 1.0})
    test_writer.add_summary(image_summary)

    print('test_label:{}, y_pred:{}'.format(test_label.argmax(), y_pred.argmax()))