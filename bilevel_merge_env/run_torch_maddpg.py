# import highway_env
import gym
import torch as th

from bilevel_trainer import Bilevel_Trainer
# from random import set_seed
from logger.utils import set_logger
from sampler import MASampler
from model import Critic, Actor, Cost
from leader_agent import LeaderAgent
from follower_agent import FollowerAgent
from indexed_replay_buffer import IndexedReplayBuffer
level_agent_num = 2


def get_leader_agent(env, agent_id):
    return LeaderAgent(
        env_specs=env.env_specs,
        policy=Actor(dim_observation=env.num_state,
                     dim_action=env.action_num),
        qf=Critic(
            dim_observation=env.num_state,
            dim_action=env.action_num
        ),
        cost_function= Cost(
            dim_observation=env.num_state,
            dim_action=env.action_num
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.agent_num,
                                          next_observation_dim = env.num_state,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        agent_id=agent_id,
    )

def get_follower_stochasitc_agent(env, agent_id):

    return FollowerAgent(
        env_specs=env.env_specs,
        policy=Actor(dim_observation=(env.num_state+env.action_num),
                     dim_action=env.action_num),
        qf=Critic(
            dim_observation=env.num_state,
            dim_action=env.action_num
        ),
        cost_function= Cost(
            dim_observation=env.num_state,
            dim_action=env.action_num
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state + env.action_num,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.agent_num,
                                          next_observation_dim = env.num_state,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        agent_id=agent_id,
    )





for seed in range(1):
    print(seed)
    # set_seed(seed)

    agent_setting = 'bilevel'
    game_name = 'merge_env'
    suffix = f'{game_name}/{agent_setting}'

    set_logger(suffix)

    env = gym.make("merge-v0")

    env.agent_num = 2
     
    env.leader_num = 1
    env.follower_num = 1
    action_num = 5
    batch_size = 64
    # training_steps = 500000
    training_steps = 1000
    exploration_step = 500
    # exploration_step = 500
    max_replay_buffer_size = 1000


    agents = []
    train_agents = []

    
    agents.append(get_leader_agent(env, 0))
    agents.append(get_follower_stochasitc_agent(env, 1))

    train_agents.append(get_leader_agent(env, 0))
    train_agents.append(get_follower_stochasitc_agent(env, 1))

    sampler = MASampler(env.agent_num, env.leader_num, env.follower_num)
    sampler.initialize(env, agents, train_agents)

    trainer = Bilevel_Trainer(
        seed=seed, env=env, agents=agents, train_agents=train_agents, sampler=sampler,
        steps=training_steps, exploration_steps=exploration_step,
        extra_experiences=['target_actions'], batch_size=batch_size
    )

    trainer.run()
    #trainer.save()