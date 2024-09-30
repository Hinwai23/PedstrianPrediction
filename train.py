import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime
import matplotlib.pyplot as plt

from dataset import DataLoader
from utils import build_model
from utils import get_lossfunc
from utils import arg
from utils import calc_prediction_error
from utils import get_coef

def train(args):
    datasets = list(range(4))

    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    # 使用参数创建一个LSTM模型
    model = build_model(args)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate)


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # 检查点保存至的目录
    checkpoint_dir = './training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    losses = []
    mean_errors = []
    final_errors = []

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states()

        for batch in range(data_loader.num_batches):
            start = time.time()

            x, y = data_loader.next_batch()

            base_pos = np.array([[e_x[0] for _ in range(len(e_x))] for e_x in x])
            x_offset = x - base_pos
            y_offset = y - base_pos


            with tf.GradientTape() as tape:
                tensor_x = tf.convert_to_tensor(x_offset, dtype=tf.float32)

                logits = model(tensor_x)

                [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                tensor_y = tf.convert_to_tensor(y_offset, dtype=tf.float32)

                [x_data, y_data] = tf.split(tensor_y, 2, -1)

                # 计算loss function
                loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                mean_error, final_error = calc_prediction_error(o_mux, o_muy, o_sx, o_sy, o_corr, tensor_y, args)

                loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.lr.assign(args.learning_rate * (args.decay_rate ** e))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss(loss)

            end = time.time()
            # Print epoch, batch, loss and time taken
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, ADE = {}, FDE = {}"
                  .format(e * data_loader.num_batches + batch,
                          args.num_epochs * data_loader.num_batches,
                          e, loss, end - start, mean_error, final_error))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        model.save_weights(checkpoint_prefix.format(epoch=e))

        losses.append(loss)
        mean_errors.append(mean_error)
        final_errors.append(final_error)

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), mean_errors)
    plt.xlabel('Epoch')
    plt.ylabel('ADE')
    plt.title('Average Displacement Error')
    plt.show()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), final_errors)
    plt.xlabel('Epoch')
    plt.ylabel('FDE')
    plt.title('Final Displacement Error')
    plt.show()



if __name__ == '__main__':
    args = arg()
    train(args)