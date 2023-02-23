import torch as th
import torch.nn as nn
import torch.nn.functional as F


# class Critic(nn.Module):
#     def __init__(self, dim_observation, dim_action):
#         super(Critic, self).__init__()
#         self.dim_observation = dim_observation
#         self.dim_action = dim_action
#         obs_dim = dim_observation
#         act_dim = self.dim_action * 2

#         self.FC1 = nn.Linear(obs_dim, 512)
#         self.FC2 = nn.Linear(512+act_dim, 256)
#         self.FC3 = nn.Linear(256, 128)
#         self.FC4 = nn.Linear(128, 1)

#     # obs: batch_size * obs_dim
#     def forward(self, obs, acts_alpha, acts_beta):
#         acts = th.cat((acts_alpha, acts_beta), 1)
#         result = F.relu(self.FC1(obs))
#         combined = th.cat([result, acts], 1)
#         result = F.relu(self.FC2(combined))
#         return self.FC4(F.relu(self.FC3(result)))
class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation
        act_dim = self.dim_action * 2

        self.FC1 = nn.Linear(obs_dim+act_dim, 32)
        self.FC2 = nn.Linear(32, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs_act):
        result = F.relu(self.FC1(obs_act))
        result = F.relu(self.FC2(result))
        return self.FC4(F.relu(self.FC3(result)))

    def cat_forward(self, obs, leader_act, follower_act,dim):
        input = th.cat((obs,leader_act,follower_act),dim=dim)
        result = F.relu(self.FC1(input))
        result = F.relu(self.FC2(result))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 32)
        self.FC2 = nn.Linear(32, 16)
        self.FC3 = nn.Linear(16, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        # result = F.gumbel_softmax(result)
        result = F.softmax(result)
        return result
    
class Cost(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Cost, self).__init__()
        self.FC1 = nn.Linear(dim_observation+dim_action*2, 32)
        self.FC2 = nn.Linear(32, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)
    
    def forward(self, obs_act):
        result = F.relu(self.FC1(obs_act))
        result = F.relu(self.FC2(result))
        result = self.FC4(F.relu(self.FC3(result)))
        return result

    def cat_forward(self, obs, leader_act, follower_act, dim):
        input = th.cat((obs,leader_act,follower_act),dim=dim)
        result = F.relu(self.FC1(input))
        result = F.relu(self.FC2(result))
        result = self.FC4(F.relu(self.FC3(result)))
        return result