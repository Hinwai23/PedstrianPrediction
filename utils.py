import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    """
    parser.add_argument('--num_layers', type=int, default=1,
                     help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    """
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    parser.add_argument('--test_dataset', type=int, default=4,
                        help='Dataset to be tested on')
    parser.add_argument('--obs_length', type=int, default=15,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=15,
                        help='Predicted length of the trajectory')

    args = parser.parse_args()

    return args


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    normx = tf.math.subtract(x, mux)
    normy = tf.math.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.math.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.math.square(tf.math.divide(normx, sx)) + tf.math.square(tf.math.divide(normy, sy)) - 2 * tf.math.divide(
        tf.math.multiply(rho, tf.math.multiply(normx, normy)), sxsy)

    negRho = 1 - tf.math.square(rho)
    # Numerator
    result = tf.math.exp(tf.math.divide(-z, 2 * negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.math.multiply(sxsy, tf.math.sqrt(negRho))
    # Final PDF calculation
    result = tf.math.divide(result, denom)

    return result

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

    epsilon = 1e-20

    result1 = -tf.math.log(tf.math.maximum(result0, epsilon))   # Numerical stability

    return tf.reduce_sum(result1)


def get_coef(output):
    z = output

    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, -1)

    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]





def get_mean_error(pred_traj, true_traj, observed_length):
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = pred_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i - observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def get_final_error(pred_traj, true_traj):
    error = np.linalg.norm(pred_traj[-1, :] - true_traj[-1, :])

    # Return the mean error
    return error


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    # 提取平均值
    mean = [mux, muy]

    # 提取协方差矩阵
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    # 从多元正态分布中确定一个点
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


# RNN
def build_model(args):
    output_size = 5
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(args.embedding_size, activation = tf.keras.activations.relu,
            batch_input_shape = [args.batch_size, None, 2]),
        tf.keras.layers.LSTM(args.rnn_size,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(output_size)
    ])

    return model



def calc_prediction_error(mux, muy, sx, sy, corr, offset_positions, args):
    traj_nums = mux.shape[0]

    pred_nums = mux.shape[1]

    mean_error = 0.0
    final_error = 0.0
    for index in range(traj_nums):
        pred_traj = np.zeros((pred_nums, 2))
        for pt_index in range(pred_nums):
            next_x, next_y = sample_gaussian_2d(mux[index][pt_index][0],
                                                muy[index][pt_index][0], sx[index][pt_index][0],
                                                sy[index][pt_index][0], corr[index][pt_index][0])

            pred_traj[pt_index][0] = next_x
            pred_traj[pt_index][1] = next_y

        mean_error += get_mean_error(pred_traj, offset_positions[index], args.obs_length)
        final_error += get_final_error(pred_traj, offset_positions[index])

    mean_error = mean_error / traj_nums
    final_error = final_error / traj_nums

    return mean_error, final_error