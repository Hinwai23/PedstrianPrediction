import os
import pickle
import numpy as np
import random

class DataLoader():

    def __init__(self, batch_size=16, seq_length=30, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        # 存储原始数据所在的数据目录的list
        self.data_dirs = ['./data/up/hs-up', './data/up/kx-up',
                         './data/down/hs-down', './data/down/kx-down',
                         './data/test']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # 定义pickle文件所在的目录
        self.data_dir = './data'

        # 接收批处理大小和序列长度参数
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 定义存储数据的cpkl文件的路径
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # 如果文件不存在或者forcePreProcess为true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # 预处理csv文件中的数据
            self.preprocess(self.used_data_dirs, data_file)

        # 从pickle文件中加载数据
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        # dataset
        for directory in data_dirs:
            # 定义数据集的csv文件的路径
            file_path = os.path.join(directory, 'pixel_pos.csv')

            print("processing {}".format(file_path))

            # 从csv文件加载数据
            # data是4 x numTrajPoints的矩阵
            # 其中每列是一个(frameId, pedId, y, x)vector
            data = np.genfromtxt(file_path, delimiter=',')

            # 获取当前数据集中行人的数量
            numPeds = np.size(np.unique(data[1, :]))

            # pedestrian
            for ped in range(1, numPeds+1):
                # 提取当前行人的轨迹
                traj = data[:, data[1, :] == ped]
                # 将其格式化为 (x, y, frameId)
                traj = traj[[3, 2, 0], :]

                # 将轨迹存到dictionary里
                all_ped_data[current_ped + ped] = traj


            dataset_indices.append(current_ped+numPeds)
            current_ped += numPeds

            print("total ped nums: {}".format(numPeds))

        # 处理后完整的数据是所有行人数据和数据集索引的元组
        complete_data = (all_ped_data, dataset_indices)
        # 将完整的数据存储到pickle文件中
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        # 从pickled文件中加载数据
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # 从pickle文件中获取行人数据
        all_ped_data = self.raw_data[0]

        # 用比seq_length长的序列(轨迹)构造数据
        self.data = []
        counter = 0

        # 对于数据中每个行人
        for ped in all_ped_data:
            # 提取轨迹
            traj = all_ped_data[ped]
            # 如果轨迹的长度大于seq_length(+2是因为需要源数据和目标数据)
            if traj.shape[1] > (self.seq_length + 2):
                # 存储(x,y)坐标
                self.data.append(traj[[0, 1], :].T)
                # 此datapoint的批次数
                counter += int(traj.shape[1] / ((self.seq_length + 2)))

        print("all ped data len: {}, seq length: {}".format(len(all_ped_data), self.seq_length))
        # 计算数据中的批(每一个batch_size)的数量
        self.num_batches = int(counter / self.batch_size)

# 训练时调用batch数据的函数
    def next_batch(self):
        # 定义当前批处理的源数据（用于训练的）和目标数据（作为原始数据对比）list
        x_batch = []
        y_batch = []
        # 对于batch中的每个序列
        for i in range(self.batch_size):
            # 提取由self.pointer指出的行人的轨迹
            traj = self.data[self.pointer]
            # 与其轨迹相对应的序列数
            n_batch = int(traj.shape[0] / (self.seq_length+2))
            # 随机抽取一个idx ??
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            # 将从idx到seq_length的轨迹数据添加到源数据和目标数据中
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))

            if random.random() < (1.0/float(n_batch)):
                # 调整 sampling probability
                # 如果这是一个较长的数据点，那么以更高的概率对该数据进行更多采样
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0


