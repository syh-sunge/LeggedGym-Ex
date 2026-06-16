# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch #张量运算
import torch.nn as nn #构建神经网络
from torch.distributions import Normal #构建高斯分布（正态分布）
from .actor_critic import get_activation

'''
Actor-Critic for Concurrent Teacher-Student architecture.
'''

class ActorCriticCTS(nn.Module):
    is_recurrent = False #表示该cts是mlp不是lstm，所以student看到和history obs被拼接成了长向量接入mlp

    def __init__(self,
                 num_actor_obs,
                 num_actions, #策略输出的关节action维度
                 num_privilege_encoder_input,
                 num_history_encoder_input,
                 num_latent_dims, #latent输出维度
                 num_critic_obs,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 privilege_encoder_hidden_dims=[256, 128],
                 history_encoder_hidden_dims=[256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super().__init__() #父类初始化

        activation = get_activation(activation) #激活函数

        mlp_input_dim_a = num_actor_obs + num_latent_dims
        # input of the critic is the concatenation of actor observation and the latent from the privilege encoder
        mlp_input_dim_c = num_critic_obs #critic obs不会主动拼接latent

        # Privilege encoder
        # 建立privilege encoder第一层：privileged obs -> linear layer -> activation
        privilege_encoder_layers = []
        privilege_encoder_layers.append(
            nn.Linear(num_privilege_encoder_input, privilege_encoder_hidden_dims[0]))
        privilege_encoder_layers.append(activation)
        # 负责继续构造后续层 最后输出num_latent_dims
        for l in range(len(privilege_encoder_hidden_dims)): # 遍历hidden dims这个列表 len()=2,所以循环l表示当前正在处理第几层
            if l == len(privilege_encoder_hidden_dims) - 1: # 如果当前层是最后一层了也就是l == 1，那么输出维度就是num_latent_dims
                privilege_encoder_layers.append(
                    nn.Linear(privilege_encoder_hidden_dims[l], num_latent_dims))
            else: # 如果当前不是最后一个hidden dim也就是 l == 0 ， 就继续连接hidden层
                privilege_encoder_layers.append(nn.Linear(
                    privilege_encoder_hidden_dims[l], privilege_encoder_hidden_dims[l + 1]))
                privilege_encoder_layers.append(activation) # 在hidden后面加激活函数
        self.privilege_encoder = nn.Sequential(*privilege_encoder_layers)

        # History encoder
        history_encoder_layers = []
        history_encoder_layers.append(
            nn.Linear(num_history_encoder_input, history_encoder_hidden_dims[0]))
        history_encoder_layers.append(activation)
        for l in range(len(history_encoder_hidden_dims)):
            if l == len(history_encoder_hidden_dims) - 1:
                history_encoder_layers.append(
                    nn.Linear(history_encoder_hidden_dims[l], num_latent_dims))
            else:
                history_encoder_layers.append(
                    nn.Linear(history_encoder_hidden_dims[l], history_encoder_hidden_dims[l + 1]))
                history_encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*history_encoder_layers)

        # Policy
        # actor网络
        # actor第一层输入是obs+latent
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1: # 如果当前层是最后一层了也就是l == 2，那么输出维度就是num_actions
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        # 把前面列表里面存号的所有网络层，按照顺序打包成一个完整的actor网络，并赋值给self.actor
        # 这表示actor网络是一个由多层linear+activation组成的mlp，最后输出num_actions维的动作均值
        # 之后只要是调用actor就会按照顺序执行这些层的计算，输入是obs+latent，输出是动作均值
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1: # 如果当前层是最后一层了也就是l == 2，那么输出维度就是1 因为critic输出一个值函数评估当前状态的好坏
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:                                # 如果当前层不是最后一层了也就是l == 0 or l == 1，那么继续连接hidden层
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        # 把前面列表里面存号的所有网络层，按照顺序打包成一个完整的critic网络，并赋值给self.critic
        self.critic = nn.Sequential(*critic_layers)

        print(f"Privilege Encoder MLP: {self.privilege_encoder}")
        print(f"History Encoder MLP: {self.history_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # 动作标准差
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    # 返回当前策略标准差 也就是self.std
    @property
    def action_std(self):
        return self.distribution.stddev

    # 计算动作分布entropy，每个维度都有一个entropy，最后求和得到总的entropy
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 根据当前网络参数和obs重新构造当前策略分布
    def update_distribution(self, observations, observation_history, privilege_observations, act_type):
        # teacher分支
        if act_type == "teacher":
            latent = self.privilege_encoder(privilege_observations)
        elif act_type == "student":
            latent = self.history_encoder(observation_history)
        else:
            raise ValueError("Invalid act_type. Must be 'teacher' or 'student'.")
        # 拼接obs和latent
        mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        # 构造高斯分布也就是强化学习中的policy，均值是actor网络输出的动作均值，标准差是self.std（可学习参数）
        self.distribution = Normal(mean, mean*0. + self.std)

    # 1.更新当前策略分布 第一行
    # 2.从分布中采样动作 第二行
    def act(self, observations, observation_history, privilege_observations, act_type, **kwargs):
        self.update_distribution(observations, observation_history, privilege_observations, act_type)
        return self.distribution.sample()

    # 计算当前分布下的log probability 也就是log pi(a|s) 这里的a是输入的动作，s是之前输入actor网络的obs+latent
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 训练采样用act() 部署/评估用act_teacher()和act_student()，不加噪声

    # 只返回均值
    def act_teacher(self, observations, privilege_observations, **kwargs):
        latent = self.privilege_encoder(privilege_observations)
        actions_mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        return actions_mean

    # 部署时只用student
    def act_student(self, observations, observation_history, **kwargs):
        latent = self.history_encoder(observation_history)
        actions_mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        return actions_mean

    # 只是计算当前value的预测，critic的更新不发生在这里
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
