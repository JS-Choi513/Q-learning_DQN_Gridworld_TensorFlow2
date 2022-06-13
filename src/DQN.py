import tensorflow as tf
import numpy as np
import pygame
from Board import Board
from Window import Window
from Player import Player
from Constants import ACTIONS, NUM_EPISODES, EPSILON
import random
import operator
import copy

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

class DQN_model(Model):
    
    def __init__(self, action_n):
        super(DQN_model, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16,activation='relu')
        self.h4 = Dense(action_n, activation='tanh')
        

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        #q = self.q(x)
        #r = tf.squeeze(q)
        return tf.squeeze(self.h4(x))

