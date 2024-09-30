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


def size(args):
    datasets = list(range(4))
    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)
    model = build_model(args)

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states()

        for batch in range(data_loader.num_batches):
            x, y = data_loader.next_batch()
            base_pos = np.array([[e_x[0] for _ in range(len(e_x))] for e_x in x])
            x_offset = x - base_pos
            y_offset = y - base_pos
            tensor_x = tf.convert_to_tensor(x_offset, dtype=tf.float32)
            print(tensor_x.shape)

if __name__ == '__main__':
    args = arg()
    size(args)