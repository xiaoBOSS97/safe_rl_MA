import torch as th
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from utils import soft_update

class FollowerAgent:
    def __init__(self,
                 env_specs,
                 policy,
                 qf,
                 cost_function,
                 replay_buffer,
                 exploration_strategy=None,
                 exploration_interval=10,
                 target_update_tau=0.01,
                 target_update_period=10,
                 gamma=1,
                 reward_scale=1.0,
                 train_sequence_length=None,
                 name='Bilevel_leader',
                 agent_id=-1
                 ):

        self._agent_id = agent_id
        self._env_specs = env_specs
        
        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space
        self._action_space = action_space
        self._exploration_strategy = exploration_strategy

        self._policy = policy
        self._qf = qf
        self._cost_function = cost_function

        self._target_policy = deepcopy(policy)
        self._target_qf = deepcopy(qf)
        self._target_cost_function = deepcopy(cost_function)

        self.replay_buffer = replay_buffer

        self._policy_optimizer = Adam(self._policy.parameters(), lr=0.001)
        self._qf_optimizer = Adam(self._qf.parameters(), lr=0.001)
        self._cost_optimizer = Adam(self._cost_function.parameters(), lr=0.001)

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._cost_scale = reward_scale
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'opponent_actions', 'target_actions']


    def get_policy_np(self, input_tensor):
        return self._policy.get_policy_np(input_tensor)

    def act(self, observation, step=None, use_target=False):
        if use_target and self._target_policy is not None:
            cur_pol = self._target_policy.forward(observation)
            cur_pol = cur_pol.detach().numpy()
            #print("target:",cur_pol)
            if cur_pol.ndim==1:
                actions = np.argmax(cur_pol)
                return [actions]
            else:
                actions = np.argmax(cur_pol,axis=1)
                return actions
        else:
            #print("no target obs:",observation)
            cur_pol = self._policy.forward(observation)
            cur_pol = cur_pol.detach().numpy()
            #print("no target:",cur_pol)
            if cur_pol.ndim==1:
                actions = np.argmax(cur_pol)
                return [actions]
            else:
                actions = np.argmax(cur_pol,axis=1)
                return actions


    def init_eval(self):
        self._exploration_status = False

    def _update_target(self):
        soft_update(self._target_qf, self._qf, self._target_update_tau)
        soft_update(self._target_cost_function, self._cost_function, self._target_update_tau)
        soft_update(self._target_policy, self._policy, self._target_update_tau)
    
    def _train(self, batch, env, agent_id, weights=None):

        critic_loss = self.critic_loss(env,
                                       agent_id,
                                       batch['observations'],
                                       batch['actions'],
                                       batch['opponent_actions'],
                                       batch['target_actions'],
                                       batch['rewards'],
                                       batch['next_observations'],
                                       batch['terminals'],
                                       weights=weights)

        cost_loss = self.cost_loss(env,
                                     agent_id,
                                     batch['observations'],
                                     batch['actions'],
                                     batch['opponent_actions'],
                                     batch['target_actions'],
                                     batch['costs'],
                                     batch['next_observations'],
                                     batch['terminals'])

        actor_loss = self.actor_loss(env, 
                                     agent_id, 
                                     batch['observations'], 
                                     batch['opponent_actions'], 
                                     weights=weights)
        self._train_step += 1

        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {
            'pg_loss': actor_loss.detach().numpy(),
            'critic_loss': critic_loss.detach().numpy(),
        }

        return losses

    def get_critic_value(self,
                         input_tensor):
        return self._qf.get_values(input_tensor)

    def critic_loss(self,
                    env, 
                    agent_id,
                    observations,
                    actions,
                    opponent_actions,
                    target_actions,
                    rewards,
                    next_observations,
                    terminals,
                    weights=None):
        """Computes the critic loss for DDPG training.
        Args:
          observations: A batch of observations.
          actions: A batch of actions.
          rewards: A batch of rewards.
          next_observations: A batch of next observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
        Returns:
          critic_loss: A scalar critic loss.
        """

        self._qf_optimizer.zero_grad()

        action_num = env.action_num
        num_state = env.num_state
        
        agent_num = target_actions.shape[1]
        batch_size = target_actions.shape[0]
        
        target_actions = th.tensor(target_actions, dtype=th.int64)
        target_actions_concat = th.zeros((batch_size, 2 * action_num))
        actions_concat = th.zeros((batch_size, 2 * action_num))
        
        
        target_actions_concat[:, :action_num] = F.one_hot(target_actions[:, agent_id], action_num)
        target_actions_concat[:, action_num:] = F.one_hot(target_actions[:, 1 - agent_id], action_num)

        opponent_actions = th.tensor(opponent_actions, dtype=th.int64)
        actions_concat[:, :action_num] = th.from_numpy(actions)
        actions_concat[:, action_num:] = F.one_hot(opponent_actions[:, 1 - agent_id], action_num)

        target_critic_input = th.zeros((batch_size, num_state + 2 * action_num))
        target_critic_input[:, :num_state] = th.from_numpy(next_observations)
        target_critic_input[:, num_state:] = target_actions_concat

        rewards = th.from_numpy(rewards.reshape(-1, 1))
        target_q_values = self._target_qf(target_critic_input)
        #print(terminals.reshape(-1, 1))
        target_q = self._reward_scale * rewards + (1 - th.from_numpy(terminals.reshape(-1, 1))) * self._gamma * target_q_values

        critic_net_input = np.hstack((observations[:, :num_state], actions_concat))
        current_q = self._qf(th.from_numpy(critic_net_input))

        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        critic_loss.backward()
        self._qf_optimizer.step()

        # if weights is not None:
        #     critic_loss = weights * critic_loss

        # critic_loss = tf.reduce_mean(critic_loss)
        # print(critic_loss)
        return critic_loss

    def cost_loss(self,
                  env, 
                  agent_id,
                  observations,
                  actions,
                  opponent_actions,
                  target_actions,
                  costs,
                  next_observations,
                  terminals):

        # init optimizer
        self._cost_optimizer.zero_grad()

        # init parameters
        action_num = env.action_num
        num_state = env.num_state
        batch_size = target_actions.shape[0]
        
        # target actions
        target_actions = th.tensor(target_actions, dtype=th.int64)
        target_actions_concat = th.zeros((batch_size, 2 * action_num))
        target_actions_concat[:, :action_num] = F.one_hot(target_actions[:, agent_id], action_num)
        target_actions_concat[:, action_num:] = F.one_hot(target_actions[:, 1 - agent_id], action_num)

        # concatnate next observation and target actions
        target_cost_input = th.zeros((batch_size, num_state + 2 * action_num))
        target_cost_input[:, :num_state] = th.from_numpy(next_observations)
        target_cost_input[:, num_state:] = target_actions_concat

        # compute target cost
        costs = th.from_numpy(costs.reshape(-1, 1))
        next_cost = self._target_cost_function(target_cost_input)
        target_cost = self._cost_scale * costs + (1 - th.from_numpy(terminals.reshape(-1, 1))) * self._gamma * next_cost

        # current observation and actions
        actions_concat = th.zeros((batch_size, 2 * action_num))
        opponent_actions = th.tensor(opponent_actions, dtype=th.int64)
        actions_concat[:, :action_num] = th.from_numpy(actions)
        actions_concat[:, action_num:] = F.one_hot(opponent_actions[:, 1 - agent_id], action_num)
        
        # current cost
        current_cost_input = np.hstack((observations[:, :num_state], actions_concat))
        current_cost = self._cost_function(th.from_numpy(current_cost_input))

        # loss optimize
        cost_loss = nn.MSELoss()(current_cost, target_cost.detach())
        cost_loss.backward()
        self._cost_optimizer.step()

        return cost_loss

    def actor_loss(self, env, agent_id, observations, opponent_actions, weights=None):
        """Computes the actor_loss for DDPG training.
        Args:
          observations: A batch of observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
          # TODO: Add an action norm regularizer.
        Returns:
          actor_loss: A scalar actor loss.
        """

        self._policy_optimizer.zero_grad()

        batch_size = observations.shape[0]

        observations = th.from_numpy(observations)

        policies = self._policy(observations)
        tot_q_values = 0
        
        actions_concat = th.zeros((batch_size, 2 * env.action_num))
        
        opponent_actions = th.tensor(opponent_actions, dtype=th.int64)
        actions_concat[:, env.action_num:] = F.one_hot(opponent_actions[:, 1 - agent_id], env.action_num)
        
        # policies (64,5)
        for action in range(policies.shape[1]):
            actions = th.full([observations.shape[0]], action)
            actions = F.one_hot(actions, self._action_space.n)
            actions_concat[:, :env.action_num] = actions
            
            q_values = self._qf(th.cat((observations[:, :env.num_state], actions_concat), 1))
            
            # if tot_q_values == None:
            #     tot_q_values = th.mul(policies[:,action:action+1] , q_values)
            # else:
            tot_q_values += th.mul(policies[:,action:action+1] , q_values)



        actor_loss = -tot_q_values.mean()
        actor_loss.backward()
        self._policy_optimizer.step()
        return actor_loss
    
    def penalty_obj(self,next_obs,leader_action,follower_action):
        return -self._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0)+\
                self._target_cost_function.cat_forward(next_obs,leader_action,follower_action,dim=0)