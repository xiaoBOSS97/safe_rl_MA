import numpy as np 
import matplotlib.pyplot as plt
seed = 160
lst = np.arange(0, 1)

reward_b_seeds = []
success_b_seeds = []
target_b_seeds = []
test_episode = 599

reward_sum_b_0 = 0
reward_sum_b_1 = 0
success_sum_b = 0
target_sum_b = 0
success_sum_b = 0

num_of_seed = 1
for seed in lst:
    reward_b_0 = np.load('./exp/exp2/reward0_BILEVEL_1x1_'+str(seed)+'.npy')
    reward_b_1 = np.load('./exp/exp2/reward1_BILEVEL_1x1_'+str(seed)+'.npy')
    success_b = np.load('./exp/exp2/success_merge_BILEVEL_1x1_'+str(seed)+'.npy')
    target_b = np.load('./exp/exp2/target_merge_BILEVEL_1x1_'+str(seed)+'.npy')

    print(success_b.shape)
    
    reward_sum_b_0 += np.sum(reward_b_0[-test_episode:])
    reward_sum_b_1 += np.sum(reward_b_1[-test_episode:])
    success_sum_b += np.sum(success_b[-test_episode:])
    target_sum_b +=np.sum(target_b[-test_episode:])

    reward_b_seeds.append(reward_b_0[-test_episode:])
    success_b_seeds.append(success_b[-test_episode:])
    target_b_seeds.append(target_b[-test_episode:])



print("mean reward for b upper agent = ", reward_sum_b_0 / (test_episode * num_of_seed))
print("mean reward for b lower agent = ", reward_sum_b_1 / (test_episode * num_of_seed))
print("mean target rate for b = ", target_sum_b / (test_episode * num_of_seed))
print("mean fail rate for b = ", 1- success_sum_b / (test_episode * num_of_seed))
print("mean untargeted rate for b = ", success_sum_b / (test_episode * num_of_seed) - target_sum_b / (test_episode * num_of_seed))


reward_seeds = []
success_seeds = []
target_seeds = []

success_seeds.append(success_b_seeds)
print(len(success_b_seeds))
target_seeds.append(target_b_seeds)



success_rate_b = []
success_rate_i = []
success_rate_m = []
success_count_b = 0
success_count_m = 0
success_count_i = 0 




errors = []
means = []

means_all = []
sums_all = []
errors_all = []

means_b = []
means_m = []
means_i = []
errors_b = []
errors_m = []
errors_i = []



num_of_algorihtm = 1
length = test_episode
mean_count = 50
reward_means = np.zeros((num_of_algorihtm, length - mean_count))
reward_errors = np.zeros((num_of_algorihtm, length - mean_count ))
success_means = np.zeros((num_of_algorihtm, length - mean_count ))
success_errors = np.zeros((num_of_algorihtm, length - mean_count))
target_means = np.zeros((num_of_algorihtm, length - mean_count ))
target_errors = np.zeros((num_of_algorihtm, length - mean_count ))
non_target_means = np.zeros((num_of_algorihtm, length - mean_count))
non_target_errors = np.zeros((num_of_algorihtm, length - mean_count))
fail_means = np.zeros((num_of_algorihtm, length - mean_count))
fail_errors = np.zeros((num_of_algorihtm, length - mean_count))


   

for i in range(num_of_algorihtm):
    for k in range(mean_count, length):
        reward_sums = []
        success_sums = []
        target_sums = []
        non_target_sums = []
        fail_sums = []
        for j in range(num_of_seed):
            success_sums.append(np.mean(success_seeds[i][j][k-mean_count:k]))
            target_sums.append(np.mean(target_seeds[i][j][k-mean_count:k]))
            non_target_sums.append(np.mean(success_seeds[i][j][k-mean_count:k]) - np.mean(target_seeds[i][j][k-mean_count:k]))
            fail_sums.append(1 - np.mean(success_seeds[i][j][k-mean_count:k]))
    
        target_means[i][k - mean_count] = np.mean(target_sums)
        target_errors[i][k - mean_count] = np.std(target_sums)
        success_means[i][k - mean_count] = np.mean(success_sums)
        success_errors[i][k - mean_count] = np.std(success_sums)   
        non_target_means[i][k - mean_count] = np.mean(non_target_sums)
        non_target_errors[i][k - mean_count] = np.std(non_target_sums)
        fail_means[i][k - mean_count] = np.mean(fail_sums)
        fail_errors[i][k - mean_count] = np.std(fail_sums)
        
print(len(reward_means[0]))

episodes = np.arange(mean_count, length)


reward_means = np.array(reward_means)
reward_errors = np.array(reward_errors)
success_means = np.array(success_means)
success_errors = np.array(success_errors)
target_means = np.array(target_means)
target_errors = np.array(target_errors)
non_target_means = np.array(non_target_means)
non_target_errors = np.array(non_target_errors)
fail_means = np.array(fail_means)
fail_errors = np.array(fail_errors)

#Reward Curve

#plt.plot(episodes, reward_means[0], label='bilevel')

'''
plt.plot(episodes, reward_means[1], label='maddpg')
plt.plot(episodes, reward_means[2], label='independent q')
plt.fill_between(episodes, reward_means[0]-reward_errors[0], reward_means[0] + reward_errors[0], alpha=0.2)
plt.fill_between(episodes, reward_means[1]-reward_errors[1], reward_means[1] + reward_errors[1], alpha=0.2)
plt.fill_between(episodes, reward_means[2]-reward_errors[2], reward_means[2] + reward_errors[2], alpha=0.2)
'''


#plt.plot(episodes, non_target_means[0], label='Non-target merge')
#plt.fill_between(episodes, fail_means[0]-fail_errors[0], fail_means[0] + fail_errors[0], alpha=0.2)
plt.subplot(1, 2, 1)
plt.ylim(0, 1)
plt.xlim(0, 700)
plt.xlabel('Episode',{'size':16})
plt.ylabel('Rate', {'size':16})
plt.tick_params(labelsize=14)
plt.fill_between(episodes, 0, non_target_means[0], color='wheat')
plt.fill_between(episodes, non_target_means[0], non_target_means[0] + fail_means[0], color='lightpink')
plt.fill_between(episodes, non_target_means[0] + fail_means[0], 1, color='lightgreen')

#plt.plot(episodes, reward_means[1], label='maddpg')
#plt.plot(episodes, reward_means[2], label='independent q')
plt.legend(['Follower go first', 'Crash', 'Leader go first'], loc=1)
plt.tight_layout()

# plt.subplot(1, 2, 2)
# plt.ylim(0, 1)
# plt.xlim(1000, 4500)
# plt.xlabel('Episode',{'size':16})
# plt.ylabel('Rate', {'size':16})
# plt.tick_params(labelsize=14)
# plt.fill_between(episodes, 0, non_target_means[1], color='wheat')
# plt.fill_between(episodes, non_target_means[1], non_target_means[1] + fail_means[1], color='lightpink')
# plt.fill_between(episodes, non_target_means[1] + fail_means[1], 1, color='lightgreen')

# #plt.plot(episodes, reward_means[1], label='maddpg')
# #plt.plot(episodes, reward_means[2], label='independent q')
# plt.legend(['Follower go first', 'Crash', 'Leader go first'], loc=1)
# plt.tight_layout()

#plt.show()
plt.savefig("exp/exp2/1x1_merge.pdf")
plt.show()
