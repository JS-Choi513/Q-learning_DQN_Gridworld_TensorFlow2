from Window import Window
from Board import Board
from Player import Player
from QLearner import QLearner
from DQN_learner import DQN_learner
import pygame
import time
from Constants import ACTIONS, NUM_EPISODES, EPSILON
import tensorflow as tf
import numpy as np
import random

pygame.init()
w = Window()
p = Player()
b = Board()
b.initCellRewards()
b.createPenaltyCells()
w.drawSurface(b, p)

prev_action = "stay"
q = DQN_learner(b)
#q.train(NUM_EPISODES)

print(b.getCellMap())
print(type(b.getCellMap()))
q.model.load_weights(".maze_solve_dqn.h5")
currNode = (p.getCurrCoords())
time.sleep(2)

def move(state, prev_action):    
    #input_state = np.reshape(tf.convert_to_tensor(state,dtype=tf.float32),(-1,2))
    #print(input_state)
    #prediction_act = tf.argmax(q.model.predict(input_state,)).numpy() 
    #action = ACTIONS[prediction_act]
    act, action =q.epsilonGreedy(state)
    p.move(action)
    #if b.isValidCell(state, action):            
    #    print("pridiction  value argmax is...",prediction_act)  
    #    p.move(action)
    #    prev_action = action
    #else:                             
    #    print("invalid action... stay")
    #    p.move(prev_action)

while(not b.isTerminalCell(currNode)):
    time.sleep(0.5)
    move(currNode,prev_action)
    
    currNode = p.getCurrCoords()
    print(currNode)
    w.colorCell(currNode, (0, 0, 255))
    pygame.display.update()
