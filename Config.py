import torch
import numpy as np
# 设置numpy的print保留小数点后四位
np.set_printoptions(precision=4, suppress=True)


class Config:
    """-----------------环境 参数-----------------"""
    # env_targets = [3600, 1800]
    episode = 2000  # 最大训练回合数
    max_step = 200  # 单回合最大步长
    # state: [-0.9940742 - 0.10870332  0.13042141], state_type: <class 'numpy.ndarray'>
    # action: [0.7554147], action_type: <class 'numpy.ndarray'>
    # reward: -9.238157286214705, reward_type: <class 'numpy.float64'>
    # done: False, done_type: <class 'bool'>,
    state_dim = 3  # 状态维度
    action_dim = 1  # 动作维度
    reward_dim = 1  # 奖励维度
    buffer_size = 256  # 经验池能够容纳的回合数大小
    batch_size = 64  # 抽取batch_size个回合的数据进行更新
    sequence_len = 10  # 抽取序列的长度
    max_ep_len = max_step
    scale = 100  # 对奖励进行等比缩小
    """--------------Transformer 参数--------------"""
    action_tanh = True
    hidden_size = 128
    n_layer = 2  # block的个数 标准的transformer 六个encoder decoder
    n_head = 2  # 注意力机制的头
    n_inner = 128
    n_position = 512
    active_func = 'relu'
    res_pdrop = 0.05
    att_pdrop = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行的位置
    train_num = 20  # 每次训练多少次
    lr = 1e-3
    momentum = 0.99


if __name__ == '__main__':

    arg = Config()
