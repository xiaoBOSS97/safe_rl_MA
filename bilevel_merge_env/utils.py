from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch as th
from torch.autograd import Variable

def add_target_actions(env, sampler, batch_n, agents, train_agents, batch_size):
    target_actions_n = []
    
    '''
      if an invalid car has not merged yet, then its action is idle until
      reaching to the merge starting point; if an invalid car has already merged, then
      we will not train our model using experience from that car
    '''
    # action taken by the leader
    for i in range(sampler.leader_num):
        #print("next", batch_n[i]['next_observations'])
        target_actions_n.append(train_agents[0].act(th.from_numpy(batch_n[i]['next_observations']), use_target=True))
        #print("leader:", batch_n[i]['next_observations'][0])
       
    target_leader_actions = th.tensor(target_actions_n)
    #print(target_leader_actions)
    #print(target_actions_n.shape)  # (leader_num, 32)
    #print(batch_n[i]['next_observations'].shape) # (32, 15)
    
    # mix observation of environment and leader action
    for i in range(sampler.leader_num, env.agent_num):
        mix_obs = th.zeros((batch_size, env.num_state + env.action_num))
        mix_obs[:, :env.num_state] = th.from_numpy(batch_n[i]['next_observations'])
        mix_obs[:, env.num_state:] = F.one_hot(target_leader_actions[0, :], env.action_num)
        #print("follower:", batch_n[i]['next_observations'][0])
        #print(type(mix_obs))
        target_actions_n.append(train_agents[1].act(mix_obs, use_target=True))

    # print(np.array(target_actions_n).shape)
    for i in range(len(agents)):
        target_actions = np.array(target_actions_n[i])
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))
        target_actions = np.concatenate((target_actions.reshape(-1, 1), opponent_target_actions), 1)
        assert target_actions.shape[0] == batch_size
        batch_n[i]['target_actions'] = target_actions
    return batch_n

def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def add_target_actions_sub(batch_n, train_agents, batch_size):
    
    '''
      if an invalid car has not merged yet, then its action is idle until
      reaching to the merge starting point; if an invalid car has already merged, then
      we will not train our model using experience from that car
    '''

    next_observations = th.from_numpy(batch_n[0]['next_observations'])
    target_actions_leader = np.zeros((next_observations.shape[0],2))
    target_actions_follower = np.zeros((next_observations.shape[0],2))
    
    for n in range(next_observations.shape[0]):
        next_obs = next_observations[n]
                                      
        lr = 0.1 # Outer learning rate
        ilr = 0.1 # Inner learning rate
        T = 10 # num iteration
        leader_action = Variable(th.randn(5), requires_grad=True)
        follower_action = Variable(th.randn(5), requires_grad=True)
        yt = th.zeros(T, 5)

        for i in range(T):
        
            # We nee to compute the total derivative of f wrt x
            ##    
            for j in range(T):
                #grad_y = th.autograd.grad(train_agents[1]._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0), follower_action, create_graph=True)[0]
                grad_y = th.autograd.grad(train_agents[1].penalty_obj(next_obs,leader_action,follower_action), follower_action, create_graph=True)[0]
                new_y = follower_action - ilr*grad_y
                follower_action = Variable(new_y, requires_grad=True)
                yt[j] = follower_action
            ###
            #alpha = -th.autograd.grad(train_agents[0]._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0), follower_action, retain_graph=True)[0]
            alpha = -th.autograd.grad(train_agents[0].penalty_obj(next_obs,leader_action,follower_action), follower_action, retain_graph=True)[0]
            gr = th.zeros_like(alpha)
            ###
            for j in range(T-1,-1,-1):
                y_tmp = Variable(yt[j], requires_grad=True)
                #grad_y, = th.autograd.grad( train_agents[1]._target_qf.cat_forward(next_obs,leader_action,y_tmp,dim=0), y_tmp, create_graph=True )
                grad_y, = th.autograd.grad(train_agents[1].penalty_obj(next_obs,leader_action,y_tmp), y_tmp, create_graph=True )
                loss = -ilr*grad_y@alpha
                aux1 = th.autograd.grad(loss, leader_action, retain_graph=True)[0]
                aux2 = th.autograd.grad(loss, y_tmp)[0]
                gr -= aux1
                alpha += aux2

            #grad_x = th.autograd.grad(train_agents[0]._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0), leader_action)[0] 
            grad_x = th.autograd.grad(train_agents[0].penalty_obj(next_obs,leader_action,follower_action), leader_action)[0] 
            ##
            leader_action = leader_action - lr*(grad_x + gr)
        leader_act = th.argmax(leader_action)
        follower_act = th.argmax(follower_action)

        target_actions_leader[n,0] = leader_act 
        target_actions_leader[n,1] = follower_act 

        target_actions_follower[n,1] = leader_act 
        target_actions_follower[n,0] = follower_act 
    
    #print(target_actions_leader)
    # print(follower_action)
    # print("....")
    return target_actions_leader, target_actions_follower

def add_target_actions_simple_grad(env, sampler, batch_n, agents, train_agents, batch_size):
    
    '''
      if an invalid car has not merged yet, then its action is idle until
      reaching to the merge starting point; if an invalid car has already merged, then
      we will not train our model using experience from that car
    '''
    next_observations = th.from_numpy(batch_n[0]['next_observations'])
    target_actions_leader = np.zeros((next_observations.shape[0],2))
    target_actions_follower = np.zeros((next_observations.shape[0],2))
    
    
    num_iter = 40
    learning_rate = 0.05
    for n in range(next_observations.shape[0]):
        next_obs = next_observations[n]
        leader_action = th.randn(5,requires_grad=True)
        follower_action = th.randn(5,requires_grad=True)
        opt_follower = th.optim.Adam([follower_action], lr=learning_rate)
        opt_leader = th.optim.Adam([leader_action], lr=learning_rate)
        for i in range(num_iter):
            for j in range(num_iter):
                # First solve the follower's problem
                opt_follower.zero_grad()
                loss_follower = -train_agents[1]._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0)
                                # +\
                                # 2*train_agents[1]._target_cost_function.cat_forward(next_obs,leader_action,follower_action,dim=0)
                loss_follower.backward()
                opt_follower.step()
            
            # Then solve the leader's problem  
            opt_leader.zero_grad()
            loss_leader = -train_agents[0]._target_qf.cat_forward(next_obs,leader_action,follower_action,dim=0)
                        #     +\
                        #    2*train_agents[0]._target_cost_function.cat_forward(next_obs,leader_action,follower_action,dim=0)
            loss_leader.backward()
            opt_leader.step()

        leader_act = th.argmax(leader_action)
        follower_act = th.argmax(follower_action)

        target_actions_leader[n,0] = leader_act 
        target_actions_leader[n,1] = follower_act 

        target_actions_follower[n,1] = leader_act 
        target_actions_follower[n,0] = follower_act 
    #print(target_actions_leader)
        
    return target_actions_leader, target_actions_follower

