"""
The trainer for multi-agent training.
"""
import pickle

from utils import *
import time
import numpy as np 

class Bilevel_Trainer:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, seed, env, agents, train_agents, sampler,
            batch_size=64,
            steps=10000,
            exploration_steps=100,
            training_interval=1,
            extra_experiences=['target_actions'],
            save_path=None,
    ):
        self.env = env
        self.seed = seed
        self.agents = agents
        self.train_agents = train_agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.exploration_steps = exploration_steps
        self.training_interval = training_interval
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path
        self.epsilon = 0.1
        self.success_rate = []

    def setup(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler

    def sample_batches(self):
        assert len(self.agents) > 1
        batches = []
        indices = self.agents[0].replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        return batches

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    def run(self):
        #print('trainer_start')
        for step in range(self.steps):
            if step < self.exploration_steps:
                 self.sampler.sample(explore=True)
                 continue
            
            if (np.random.uniform(0, 1) < self.env.epsilon):
                self.sampler.sample(explore=True)
            else:
                self.sampler.sample()
            
            
            
            batches = self.sample_batches()
        
            
            for extra_experience in self.extra_experiences:

                if extra_experience == 'target_actions':
                    batches = add_target_actions(self.env, self.sampler, batches, self.agents, self.train_agents, self.batch_size)
                    #batches[0]['target_actions'], batches[1]['target_actions'] = add_target_actions_sub(batches, self.train_agents, self.batch_size)
                    #batches[0]['target_actions'], batches[1]['target_actions'] = add_target_actions_simple_grad(self.env, self.sampler, batches, self.agents, self.train_agents, self.batch_size)
            agents_losses = []

            if step % self.training_interval == 0:
                
                for agent, batch in zip(self.agents, batches):
                    if agent._agent_id < self.sampler.leader_num:
                        agent_losses = self.train_agents[0]._train(batch, self.env, agent._agent_id)
                    else:
                        agent_losses = self.train_agents[1]._train(batch, self.env, agent._agent_id)
                    agents_losses.append(agent_losses)

            
            if step % 500 == 0 and step > 0:
                self.env.epsilon *= 0.9

            
            if self.env.merge_count % 100 == 0:
                np.save('./exp/exp2/reward0_BILEVEL_1x1_'+str(self.seed)+'.npy', self.env.episodes_reward_0)
                np.save('./exp/exp2/reward1_BILEVEL_1x1_'+str(self.seed)+'.npy', self.env.episodes_reward_1)                
                np.save('./exp/exp2/success_merge_BILEVEL_1x1_'+str(self.seed)+'.npy', self.env.episode_merge_record)
                np.save('./exp/exp2/target_merge_BILEVEL_1x1_'+str(self.seed)+'.npy', self.env.episode_target_merge_record) 
            
            
            if self.env.merge_count == 8001:
                break
            

            
    def save(self):
        save_path = './models/agents_bilevel_1x1_'+str(self.seed)+'.pickle'
        if self.save_path is None:
            self.save_path = './models/agents_bilevel_1x1.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(self.train_agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, restore_path):
        with open(restore_path, 'rb') as f:
            self.train_agents = pickle.load(f)

    def resume(self):
        pass

    def log_diagnostics(self):
        pass