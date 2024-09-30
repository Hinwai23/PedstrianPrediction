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


    visual_trajectories(pred_trajs[14], truth_trajs[14],
                        pred_trajs[4], truth_trajs[4],
                        pred_trajs[1], truth_trajs[1],
                        pred_trajs[2], truth_trajs[2],
                        pred_trajs[7], truth_trajs[7],
                        pred_trajs[11], truth_trajs[11],
                        pred_trajs[24], truth_trajs[24],
                        pred_trajs[13], truth_trajs[13],)






def visual_trajectories(pred_traj, true_traj,pred_traj_1, true_traj_1,pred_traj_2, true_traj_2,pred_traj_3, true_traj_3, pred_traj_4, true_traj_4, pred_traj_5, true_traj_5,pred_traj_6, true_traj_6,pred_traj_7, true_traj_7):
    fig_width = 8
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.1, 0.51, 0.05))
    ax.set_yticks(np.arange(-0.1, 0.51, 0.05))
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.1, 0.5)

# 左右
    plt.plot(-0.1,-0.1,color='Black', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Black', markerfacecolor='Black',
             label='True Trajectory')

    plt.plot(-0.1,-0.1,color='Black', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Black', markerfacecolor='Black',
             label='Predicted Trajectory')

    plt.plot(true_traj[:, 0] + 0.35, true_traj[:, 1], color='Orange', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Orange', markerfacecolor='Orange',
             )

    plt.plot(pred_traj[:, 0] + 0.35, pred_traj[:, 1], color='Orange', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Orange', markerfacecolor='Orange',
             )

    plt.plot(true_traj_1[:, 0] + 0.35, true_traj_1[:, 1] + 0.015, color='Blue', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Blue', markerfacecolor='Blue',
             )

    plt.plot(pred_traj_1[:, 0] + 0.35, pred_traj_1[:, 1] + 0.015, color='Blue', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Blue', markerfacecolor='Blue',
             )

    plt.plot(true_traj_2[:, 0] + 0.05, true_traj_2[:, 1], color='Purple', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Purple', markerfacecolor='Purple',
             )

    plt.plot(pred_traj_2[:, 0] + 0.05, pred_traj_2[:, 1], color='Purple', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Purple', markerfacecolor='Purple',
             )

    plt.plot(true_traj_3[:, 0] + 0.05, true_traj_3[:, 1] + 0.01, color='Red', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Red', markerfacecolor='Red',
             )

    plt.plot(pred_traj_3[:, 0] + 0.05, pred_traj_3[:, 1] + 0.01, color='Red', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Red', markerfacecolor='Red',
             )

# 上下

    plt.plot(true_traj_4[:, 0] + 0.05, true_traj_4[:, 1] + 0.35 , color='Green', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Green', markerfacecolor='Green',
             )

    plt.plot(pred_traj_4[:, 0] + 0.05, pred_traj_4[:, 1] + 0.35, color='Green', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Green', markerfacecolor='Green',
             )

    plt.plot(true_traj_5[:, 0] + 0.05, true_traj_5[:, 1] + 0.36, color='Pink', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Pink', markerfacecolor='Pink',
             )

    plt.plot(pred_traj_5[:, 0] + 0.05, pred_traj_5[:, 1] + 0.36, color='Pink', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Pink', markerfacecolor='Pink',
             )

    plt.plot(true_traj_6[:, 0] + 0.05, true_traj_6[:, 1] + 0.34, color='Brown', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Brown', markerfacecolor='Brown',
             )

    plt.plot(pred_traj_6[:, 0] + 0.05, pred_traj_6[:, 1] + 0.34, color='Brown', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Brown', markerfacecolor='Brown',
             )

    plt.plot(true_traj_7[:, 0] + 0.32, true_traj_7[:, 1] + 0.36, color='Gray', linestyle='-', linewidth=2.2,
             marker='p', markersize=2.1, markeredgecolor='Gray', markerfacecolor='Gray',
             )

    plt.plot(pred_traj_7[:, 0] + 0.32, pred_traj_7[:, 1] + 0.36, color='Gray', linestyle='dotted', linewidth=2.1,
             marker='p', markersize=2, markeredgecolor='Gray', markerfacecolor='Gray',
             )



    plt.title('Pedestrian Trajectory Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    visual()