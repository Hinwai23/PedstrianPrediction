import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime


from dataset import DataLoader
from utils import build_model
from utils import get_mean_error
from utils import arg
from utils import get_final_error
from utils import get_coef
from utils import sample_gaussian_2d



def test(args):
    checkpoint_dir = './training_checkpoints'

    dataset = [args.test_dataset]

    # 初始化data_loader对象以获取长度为obs_length+pred_length的序列
    data_loader = DataLoader(1, args.pred_length + args.obs_length, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    tf.train.latest_checkpoint(checkpoint_dir)

    args.batch_size = 1

    test_model = build_model(args)  # Model(args)

    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    test_model.build(tf.TensorShape([1, None, 2]))

    # Maintain the total_error until now
    total_error = 0
    counter = 0
    final_error = 0.0

    truth_trajs = []
    pred_trajs = []
    gauss_params = []

    for b in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        x, y = data_loader.next_batch()

        base_pos = np.array([[e_x[0] for _ in range(len(e_x))] for e_x in x])
        x = x - base_pos

        # The observed part of the trajectory
        obs_observed_traj = x[0][:args.obs_length]
        obs_observed_traj = tf.expand_dims(obs_observed_traj, 0)

        complete_traj = x[0][:args.obs_length]

        test_model.reset_states()

        # test_model.initial_state = None
        gauss_param = np.array([])

        for idx in range(args.pred_length):
            tensor_x = tf.convert_to_tensor(obs_observed_traj)

            logits = test_model(tensor_x)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

            next_x, next_y = sample_gaussian_2d(o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0],
                                                o_corr[0][-1][0])

            obs_observed_traj = tf.expand_dims([[next_x, next_y]], 0)

            if len(gauss_param) <= 0:
                gauss_param = np.array(
                    [o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])
            else:
                gauss_param = np.vstack(
                    (gauss_param, [o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]]))

            complete_traj = np.vstack((complete_traj, [next_x, next_y]))

        total_error += get_mean_error(complete_traj + base_pos[0], x[0] + base_pos[0], args.obs_length)
        final_error += get_final_error(complete_traj + base_pos[0], x[0] + base_pos[0])
        # total_error += get_mean_error(complete_traj, x[0], args.obs_length)
        # final_error += get_final_error(complete_traj, x[0])

        pred_trajs.append(complete_traj)
        truth_trajs.append(x[0])
        gauss_params.append(gauss_param)

        print("Processed trajectory number: {} out of {} trajectories".format(b, data_loader.num_batches))

    # Print the mean error across all the batches
    print("Total ADE of the model is {}".format(total_error / data_loader.num_batches))
    print("Total FDE error of the model is {}".format(final_error / data_loader.num_batches))

    data_file = "./pred_results.pkl"
    f = open(data_file, "wb")
    pickle.dump([pred_trajs, truth_trajs, gauss_params], f)
    f.close()

if __name__ == '__main__':
    args = arg()
    test(args)