import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import multivariate_normal


def visual():
    data_file = "./pred_results.pkl"

    f = open(data_file, "rb")
    visual_data = pickle.load(f)
    f.close()

    pred_trajs = visual_data[0]
    truth_trajs = visual_data[1]
    gauss_params = visual_data[2]

    traj_num = len(pred_trajs)

    for index in range(traj_num):
        visual_trajectories(pred_trajs[index], truth_trajs[index])
"""
    for index in range(traj_num):
        visual_gaussian(gauss_params[index])
"""





def visual_trajectories(pred_traj, true_traj):
    fig_width = 8
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
    ax.set_yticks(np.arange(-0.5, 0.5, 0.1))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    plt.plot(true_traj[:, 0], true_traj[:, 1], color='Blue', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Blue', markerfacecolor='Blue',
             label='True Trajectory')

    plt.plot(pred_traj[:, 0], pred_traj[:, 1], color='Purple', linestyle='-.', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Purple', markerfacecolor='Purple',
             label='Predicted Trajectory')

    plt.title('Pedestrian Trajectory Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show()


def visual_gaussian(gauss_params):
    fig_width = 8
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))



    plt.plot(gauss_params[:, 0], color='Blue', linestyle='-', linewidth=1,
             marker='p', markersize=1, markeredgecolor='Blue', markerfacecolor='Blue',
             label='mux')
    plt.plot(gauss_params[:, 1],  color='Red', linestyle='-', linewidth=1,
             marker='p', markersize=1, markeredgecolor='Red', markerfacecolor='Rede',
             label='muy')
    plt.plot(gauss_params[:, 2],  color='Yellow', linestyle='-', linewidth=1,
             marker='p', markersize=1, markeredgecolor='Yellow', markerfacecolor='Yellow',
             label='sx')
    plt.plot(gauss_params[:, 3],  color='Green', linestyle='-', linewidth=1,
             marker='p', markersize=1, markeredgecolor='Green', markerfacecolor='Green',
             label='sy')
    plt.plot(gauss_params[:, 4],  color='Purple', linestyle='-', linewidth=1,
             marker='p', markersize=1, markeredgecolor='Purple', markerfacecolor='Purple',
             label='corr')




    plt.show()


if __name__ == '__main__':
    visual()
