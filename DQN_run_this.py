from cmath import pi
from lib2to3 import refactor
from DQN_field_env import Flow_Field, filename
from DQN_RL_brain import DeepQNetwork
from DQN_utils import calculate_reward_from_raw_state, transform_raw_state
import torch
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt


# 只要最原始的状态,不要transform
# raw_state = (前一刻的三点坐标，目标点坐标，质心坐标，当前的三点坐标，目标点坐标，质心坐标,前一刻的phase,当前的phase)
# 经过transform后，变成12维的状态

MAX_EPISODE = 40
scale_factor = torch.tensor([10,10,20,20,10,10, 30,30, 1, 1/pi, 6/pi, 50,25, 1e4, 1/4])
scale_factor = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
def postProcess():
    with open(os.path.join('viz_IB2d', 'dumps.visit'), "r") as f:
        lines=f.readlines()
        first = lines[0]
        second = lines[1]
        step_size=int(lines[1][11:second.index('/')])-int(lines[0][11:first.index('/')])
        temp = lines[-1]
        final_step=int(lines[-1][11:temp.index('/')])
    pointer=0
    with open(os.path.join('viz_IB2d', 'dumps.visit'), "w+") as f:
        while pointer<=final_step:
            numstr=str(pointer)
            while len(numstr)<5:
                numstr='0'+numstr
            f.write("visit_dump."+numstr+"/summary.samrai\n")
            pointer+=step_size
    with open(os.path.join('viz_IB2d', 'lag_data.visit'), "r") as f:
        lines=f.readlines()
        step_size=int(lines[1][15:21])-int(lines[0][15:21])
        final_step=int(lines[-1][15:21])
    pointer=0
    with open(os.path.join('viz_IB2d', 'lag_data.visit'), "w+") as f:
        while pointer<=final_step:
            numstr=str(pointer)
            while len(numstr)<6:
                numstr='0'+numstr
            f.write("lag_data.cycle_"+numstr+"/lag_data.cycle_"+numstr+".summary.silo\n")
            pointer+=step_size

def calculate_sequence_obs(single_state, n, current_step, RL):
    pointer = current_step
    res = []
    res.append(single_state)
    while len(RL.memory.buffer)>0 and pointer>=0 and len(res)<n and 1-pointer+current_step<=len(RL.memory.buffer) :
        temp = (RL.memory.buffer[-1+pointer-current_step])[0]
        res.append(transform_raw_state(temp))
        pointer-=1
    while len(res)<n:
        temp = res[-1].clone()
        temp[-1] = (temp[-1].item() - 1 + 4)%4
        res.append(temp)
    return torch.stack(res)

####################################################################################
# Policy net (pi_theta)
import torch.nn as nn
from torch.nn import functional as F
class PolicyNet(nn.Module):
    def __init__(self, input_dim = 15, output_dim = 4, hidden_dim=128):
        super().__init__()
        self.pNet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, s):
        outs = self.pNet(s)
        return outs
class categorical:
    def __init__(self, s, pi_model):
        logits = pi_model(s)
        self._prob = F.softmax(logits, dim=-1)
        self._logp = torch.log(self._prob)
    def prob(self):
        return self._prob
    def logp(self):
        return self._logp
def select_action(pi_model, s, det=False):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float64)
        logits = pi_model(s_batch)
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1)
        a = a.squeeze(dim=0).item()
        if det==True:
            a = torch.argmax(probs).item()
        return a
####################################################################################




def start_swim(RL, times:int, train_flag):
    action_his = open('action_history.txt', 'w+')
    state_log = open("state_log.txt","w+")
    pi_model = PolicyNet().double()
    pi_model.load_state_dict(torch.load('SAC_pi_model.pth', map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    with open('total_reward.txt', 'a') as f:
        for episode in range(times):
            # 初始环境
            env = Flow_Field()
            #=============================随机给定一个目标点========================
            #init_target = np.array([2.0, 1.2])
            #init_target = np.array([1.6, 1.6])
            init_target = np.array([2.2,2.6])

            #init_target[0] = 0.8 + 0.15 * random.randint(-4,4)
            #init_target[1] = 0.8 + 0.075 * random.randint(-4,4)
            #=====================================================================
            # 初始物体状态和环境
            #env.initialize_test(init_target)
            observation = torch.tensor(env.initialize(init_target, autoPilot=True), dtype=torch.float64)
            #env.SenseSurroundingAreaAndChangeCourse()
        
            done = 0
            step = 0  # 记录步数
            RL.steps_done = step
            total_reward = 0
            last_action = -1

            if env.target_traj is None:
                print("no maze info!")
                #done = 1
            #done = 1
            #=====================================================================

            while not done:
                # all the state/observation use the raw form, only transformed it into 12d vector when choosing action
                # RL choose action based on observation
                # 输入的形状:
                # state/next_state: [1, n_observation]
                # action: [1,1]
                # reward: [1]
                # done: [1]
                print('==============current_time = ', env.currentTime, 'observation shape= ', observation.shape, '==========\n')
                print('\n++++++++++++++++++++++++++ 当前目标点 = ',env.targetPoint,'+++++++++++++++++++++++++\n')
                actual_obs = (transform_raw_state(observation)*scale_factor).unsqueeze(0)
                action_num = RL.select_action(actual_obs, not train_flag)
                action_num = select_action(pi_model, actual_obs) # SAC method

                robs = (actual_obs.squeeze())/scale_factor
                diameter = ((robs[0].item()-robs[4].item())**2+(robs[1].item()-robs[5].item())**2)**0.5
                if diameter>0.15:
                    if step%4==0:
                        action_num=0
                    else:
                        action_num=3
                if diameter<0.03:
                    if step%4==0:
                        action_num=3
                    else:
                        action_num=0
                #if step<40:
                #    action_num = 0
                tempact = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,3,0,0, 0,0,0,0, 0,0,0,0]
                #if step<len(tempact):
                #    action_num = tempact[step]

                action_reg = 1
                if action_reg == 1:
                    # action regulation
                    temp = step % 4
                    if temp == 1 or temp == 3:
                        if last_action == 0:
                            action_num = 0
                        else:
                            action_num = 3
                last_action = action_num

                action = env.action_space[action_num]

                print('========action = ', action, '==========\n')
                try:
                    action_num = action_num.item()
                except:
                    pass
                action_his.write(str(action_num) +'\n')
                for jj in range(15):
                    state_log.write(str(actual_obs.squeeze()[jj].item()) + ', ')
                state_log.write('\n'+str(action_num)+'\n\n')

                # RL take action and get next observation and reward
                #print(action)
                observation_, reward, done = env.step(action)
                print(env.pre_targetPoint, env.targetPoint)
                
                total_reward += reward
                
                # 输入的形状:
                # state/next_state: [1, n_observation]
                # action: [1,1]
                # reward: [1]
                # done: [1]
                s = observation.unsqueeze(0).to(RL.device)
                s_ = observation_.unsqueeze(0).to(RL.device)
                a = torch.tensor([[action_num]], device = RL.device)
                r = torch.tensor([reward], dtype=torch.float64, device=RL.device)
                RL.memory.push(s, a, s_, r, torch.tensor([done], device=RL.device))
                print(observation.shape)
                if train_flag==True:
                    s = transform_raw_state(observation).unsqueeze(0).to(RL.device)
                    s_ = transform_raw_state(observation_).unsqueeze(0).to(RL.device)
                    RL.memory.transformed_push(s, a, s_, r, torch.tensor([done], device=RL.device))
                # swap observation
                observation = observation_
                # train
                if train_flag==True:
                    for _ in range(200):
                        # Perform one step of the optimization (on the policy network)
                        RL.optimize_model(both_run_and_train=True)
                        # Soft update of the target network's weights
                        RL.update_network()

                #####################################################################
                # moving target
                move_flag = -2
                # 把目标点设为当前所在在block的下一个block的中心
                if move_flag==-1:
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.calculateTargetBasedOnPredefinedTraj()   
                if move_flag==-2 and step%4 in [0,2] and step>0:
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.SenseSurroundingAreaAndChangeCourse()
                # right trace
                if move_flag == 2:
                    t = step*0.5
                    fff = 1/240 #1/120 #1/240 #1/120  #1/180  #1/240  #240 180
                    center_x = 0.8
                    center_y = 0.8  #0.7
                    theta = np.arctan(abs(center_y-env.initTarget[1])/abs(center_x-env.initTarget[0]))
                    radius = ((env.initTarget[1]-center_y)**2 + (env.initTarget[0]-center_x)**2)**0.5
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.targetPoint[0] = center_x + radius * np.cos(theta + 2*pi*fff*t)
                    env.targetPoint[1] = center_y + radius * np.sin(theta + 2*pi*fff*t)
                # 向右走的双圆环
                if move_flag == 4:
                    t = step*0.5
                    fff= 1/90 #1/90 #1/120
                    TTT=1/fff
                    radius = 0.3 #0.25
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    mag = 0 #0.03
                    if t<TTT:
                        env.targetPoint[0] = 0.8 + radius - radius*np.cos(2*pi*fff*t) + mag*random.random()
                        env.targetPoint[1] = 0.8 + radius*np.sin(2*pi*fff*t) + mag*random.random()
                    elif t<2*TTT:
                        env.targetPoint[0] = 0.8 - radius + radius*np.cos(2*pi*fff*(t-TTT)) + mag*random.random()
                        env.targetPoint[1] = 0.8 + radius*np.sin(2*pi*fff*(t-TTT)) + mag*random.random()
                    if t>2*TTT and abs(observation[8].item()) < 0.02:
                        done = 1
                # 向左走的双圆环
                if move_flag == 5:
                    t = step*0.5
                    fff=1/120
                    TTT=1/fff
                    radius = 0.25
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    if t<TTT:
                        env.targetPoint[0] = 0.8 - radius + radius*np.cos(2*pi*fff*t)
                        env.targetPoint[1] = 0.8 + radius*np.sin(2*pi*fff*t)
                    elif t<2*TTT:
                        env.targetPoint[0] = 0.8 + radius - radius*np.cos(2*pi*fff*(t-TTT))
                        env.targetPoint[1] = 0.8 + radius*np.sin(2*pi*fff*(t-TTT))
                    if t>2*TTT and abs(observation[8].item()) < 0.02:
                        done = 1
                #####################################################################
                step += 1
                print('episode = ', episode, ' curren_time = ', env.currentTime, 'current total_reward = ', total_reward, '\n')
                print('\n=============================================================================================\n')
                print('\n==================================当前质心坐标:',env.massCenter,'==================================\n')
                print('\n==================================当前observation:',observation.shape,'==================================\n')
                print('\n==================================当前与目标距离:',np.linalg.norm(env.massCenter - env.targetPoint),'==================================\n')
                print('\n=============================================================================================\n')
                
            print('=========================Episode ', episode, ' Total Reward = ', total_reward, '================================\n')
            f.write(str(total_reward) + '\n')
            print("当前质心坐标",env.massCenter, "当前与目标距离",np.linalg.norm(env.massCenter - env.targetPoint),'\n')
            env.plot_traj()
            action_his.write('\n\n')
            state_log.write('\n\n')
            #plot_total_reward()
    
    # 训练结束
    print('train finished!\n')
    postProcess()
    action_his.close()
    state_log.close()


def start_mix_train(steps):
    RL=DeepQNetwork(lr=1e-2)
    RL.load_in(load_in_filename='../new_phase_base_refine_train_newReward.txt', transform_flag=False)
    RL2 = DeepQNetwork(lr=1e-4)
    for i in range(steps):
        RL.optimize_model()
        if i!=0 and i%1e3==0 and np.mean(np.array(RL.loss_history[-1000:]))<0.1:
            RL2.policy_net.load_state_dict(torch.load('DQN_policy_net'))
            RL2.target_net.load_state_dict(RL2.policy_net.state_dict())
            RL2.pretrain_model(expert_set='../expert_LeftAndRightFix_righttrace_pretrain.txt')
            RL.policy_net.load_state_dict(torch.load('DQN_policy_net'))
            RL.target_net.load_state_dict(RL.policy_net.state_dict())
    RL2.pretrain_model(expert_set='../expert_LeftAndRightFix_righttrace_pretrain.txt')
    RL.plot_loss()

def load_in_raw_and_transform_with_shifted_target(shifts):
    for shift in shifts:
        RL = DeepQNetwork()
        RL.load_in(transform_flag=True, shift=shift)
        RL.write_out()

def plot_total_reward(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    total_reward = np.loadtxt("total_reward.txt")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(total_reward)
    #plt.show()
    



if __name__ == "__main__":
    # 强化学习推进策略
    train_or_run = 1

    if train_or_run == 0:
       RL = DeepQNetwork()
       #RL.offline_train(100000) # read in raw state
       RL.offline_train(1000000, data_filename='../new_phase_base_refine_train_newReward.txt', transform_flag=False)
       filename = "state_log_right_phase_standard_refine.txt"
       RL.evaluate_network(filename)
       filename = "state_log_phase_righttrace_refine.txt"
       RL.evaluate_network(filename)
       filename = "state_log_phase_doubleCircle_refine.txt"
       RL.evaluate_network(filename)
    elif train_or_run == 1:
        for _ in range(1):
            learn_flag = False
            RL = DeepQNetwork()
            if learn_flag == True:
                try:
                    RL.load_in(transform_flag=True)
                except:
                    pass
            n = len(RL.memory)
            #print('load in experience tuple:',n,'\n')
            start_swim(RL,1,learn_flag)
            if learn_flag == True:
                RL.write_out(n)
        #plot_total_reward(show_result=True)
        #plt.show()
    elif train_or_run ==-1:
        RL = DeepQNetwork()
        RL.load_buffer()
        n = len(RL.memory.buffer)
        print('load in experience tuple:',n,'\n')
        RL.write_out_buffer(0)

    elif train_or_run ==-2:
        RL = DeepQNetwork()
        RL.pretrain_model()
    
    elif train_or_run == -4:
        start_mix_train(100000)
        RL = DeepQNetwork()
        filename = "state_log_right_phase_standard_refine.txt"
        RL.evaluate_network(filename)
        filename = "state_log_phase_righttrace_refine.txt"
        RL.evaluate_network(filename)
        filename = "state_log_phase_doubleCircle_refine.txt"
        RL.evaluate_network(filename)

   
            
  

    
    
    

    
    