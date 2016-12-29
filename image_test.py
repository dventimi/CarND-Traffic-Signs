import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

image_loader = \
        tf.image.resize_images(
            tf.image.decode_jpeg(
                tf.WholeFileReader().read(
                    tf.train.string_input_producer(
                        tf.train.match_filenames_once("./images/*.jpg")))[1]),
            [32, 32], method=1)

# plt.ion()
# fig = plt.figure()
# plt.subplots_adjust(wspace=0.001, hspace=0.001)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images = np.array([
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader),
        sess.run(image_loader)])
    coord.request_stop()
    coord.join(threads)
    # fig.add_subplot(4, 3, 1)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 2)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 3)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 4)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 5)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 6)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 7)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 8)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 9)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 10)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 11)
    # plt.imshow(sess.run(image), interpolation='nearest')
    # fig.add_subplot(4, 3, 12)
    # plt.imshow(sess.run(image), interpolation='nearest')
