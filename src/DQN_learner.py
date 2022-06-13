import pygame
from torch import choose_qparams_optimized, dtype
from Board import Board
from Window import Window
from Player import Player
from Constants import ACTIONS, NUM_EPISODES, EPSILON
import random
import matplotlib.pyplot as plt
import operator
import copy
from DQN2 import DQN_model
import tensorflow as tf
import numpy as np 
from replaybuffer_copy import ReplayBuffer
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import keras
class DQN_learner(object):
    def __init__(self, env, b=Board()):

        ## hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 2000
        self.DQN_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01
        self.board = b

        ## create Q networks
        
        self.model = self.build_model()
        
        #print(self.model.summary())
        
        
        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)        
        # save the results
        self.save_epi_reward = []
        self.discaount = 0.9
        self.alpha = 0.9
        self.currState = (0,0)
        self.movable_vec = [[-1,0],[1,0],[0,1],[0,-1]] # 'left', 'right', 'up', 'down'
    # model set is wrong,, model have to predict the maximize the reward
    # input = state, action

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(2,2), activation='tanh'))
        model.add(Flatten())
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(4, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.DQN_LEARNING_RATE))
        return model

    def choose_best_action(self, state, str_movables):
        best_actions = []
        max_act_value = -10
        movables_val = self.get_movables(state,str_movables)
        for a in movables_val:
            np_action = np.array([[state, a]])
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [a,]
                which_act = np.array(state) - np.array(a)
                str_act = self.get_str_act(which_act)
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
                which_act = np.array(state) - np.array(a)
                str_act = self.get_str_act(which_act)
        return random.choice(best_actions), str_act
        

    def get_str_act(self, act_vec):
        #print("sddsdsdsds",act_vec)
        if np.array_equal(np.array(self.movable_vec[0]), act_vec):
            return "right"
        if np.array_equal(np.array(self.movable_vec[1]), act_vec):
            return "left"
        if np.array_equal(np.array(self.movable_vec[2]), act_vec):
            return "down"
        if np.array_equal(np.array(self.movable_vec[3]), act_vec):
            return "up"                                
       
    def epsilonGreedy(self, state):
        randInt = random.randint(1,11)
        validActions = list(filter(lambda action: self.board.isValidCell(state, action), ACTIONS))        
        if randInt <= EPSILON:     

            rnd_action = random.choice(self.get_movables(state,validActions))
            return rnd_action, random.choice(validActions)
        else: #just return max action value that regardless of valid cell        
            action, str_action = self.choose_best_action(state, validActions)            
            return action, str_action 



    # coordinate list, string list, label list
    #@tf.function
    def dqn_learn(self,x,y):        
        self.model.fit(x,y, epochs=1, verbose=0)

    ## transfer actor weights to target actor with a tau
 

    def td_target(self, rewards, target_action, is_terminals):
        y_k = target_action # action of next_state 
        #print("target action......",y_k)
        #print("target action length......",len(y_k))
        #print("REWARD......", rewards)
        for i in range(len(target_action)): # number of batch
            if is_terminals[i]:
                y_k[i] = tf.constant(rewards[i], dtype=tf.float32)
            else:
                
                y_k[i] = tf.constant(rewards[i] + self.GAMMA * target_action[i], dtype=tf.float32)
                
                #print("NOt TERMINAL REWARD......",y_k[i])
        #print("y_k......",y_k)                
        return np.array(y_k)

    def get_movables(self, state, movable_action):
        movables = []
        state = list(state)
        for action in movable_action:
            if action == 'left':
                y = state[1]
                x = state[0]-1                
            elif action == 'right':
                y = state[1]
                x = state[0]+1
            elif action == 'up':
                y = state[1]+1
                x = state[0]
            elif action == 'down':
                y = state[1]-1
                x = state[0]
            movables.append([y,x])
        return movables        
    


    def get_action(self, action):
        
        if action == 'left':
            return self.movable_vec[0]
        
        elif action == 'right':
            return self.movable_vec[1]

        elif action == 'up':
            return self.movable_vec[2]
  
        elif action == 'down':
            return self.movable_vec[3]



    def train(self, max_episode_num):
            # initial transfer model weights to target model network
            count = 0
            times =  500
            for ep in range(int(max_episode_num)):
                count+=1
                time, episode_reward, is_terminal = 0, 0, False
                state = self.currState
                state = (0,0)

                for time in range(times):
                    if self.board.isTerminalCell(state) or time ==(times-1):
                        break                    
                    print("TERMINAL?",self.board.isTerminalCell(state))                 
                    action, str_action = self.epsilonGreedy(state)# output: string
                    next_state = state
                     
                    next_state = self.board.getCellAfterAction(state, str_action)# output: coordinate                      
                    reward = tf.constant(self.board.getCellValue(state), dtype=tf.float32)
                    is_terminal = self.board.isTerminalCell(next_state)# boolean 
                    print("is_termianae",is_terminal)
                    next_movables = list(filter(lambda action: self.board.isValidCell(next_state, action), ACTIONS))        
                    self.buffer.add_buffer(state, action, reward, next_state,self.get_movables(state,next_movables), is_terminal)
                    X = []
                    Y = []
                    if self.buffer.buffer_count() > 512:  # start train after buffer has some amounts
                        if self.EPSILON > self.EPSILON_MIN:
                            self.EPSILON *= self.EPSILON_DECAY
                                                    
                        states, action, rewards, next_states, next_movables, is_terminals = self.buffer.sample_batch(self.BATCH_SIZE)
                        #print("isterminals",is_terminals)
                        #print("isterminals",is_terminals.shape)
                        for i in range(self.BATCH_SIZE):
                            input_action = [states[i], np.array(action[i])]
                            
                            if(is_terminals[i]):
                                target_f = rewards[i]
                            else:
                                next_rewards = []
                                for next_action in next_movables[i]:
                                    np_next_s_a = np.array([[next_states[i],next_action]])
                                   # print("dsdsdsdsdsdsdsds",np_next_s_a.shape)
                                    next_rewards.append(self.model.predict(np_next_s_a))
                                np_next_reward_max = np.amax(np.array(next_rewards))
                                target_f = rewards[i] + self.GAMMA * np_next_reward_max
                            X.append(input_action)
                            Y.append(target_f)
                        np_X = np.array(X)
                        np_Y = np.array([Y]).T
                        if self.EPSILON > self.EPSILON_MIN:
                            self.EPSILON *= self.EPSILON_DECAY                            
                        self.dqn_learn(np_X,np_Y)
                        #self.update_target_network(self.TAU)
                    # update current state
                    state = next_state
                    print("State....",state)
                    print("Reward",reward)
                    episode_reward += reward
                    time += 1
                ## display rewards every episode
                print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
                self.save_epi_reward.append(episode_reward)
                ## save weights every episode
                self.model.save_weights(".maze_solve_dqn.h5")
            np.savetxt('.maze_solve_reward.txt', self.save_epi_reward)                
