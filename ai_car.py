# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:59:39 2020

@author: Aditya
"""

import numpy as np
import torch
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F # Google loss because that improves convergence and the loss is contained in this functional submodule 
import torch.optim as optim #optimizer
import torch.autograd as autograd
from torch.autograd import Variable

#Creaating architectur of neural network
class Network(nn.Module):
    #basic argument: number of input neurons & number of output neurons 
    #5 inputs: 3 signals; orientation; minus orientation= track of goal
    #3 output actions: forward;right;left
    def __init__(self,input_size=5, nb_action=3):
        super(Network,self).__init__()
        self.input_siz=input_size
        self.nb_action=nb_action
        self.fc1=nn.Linear(input_size,30)
        self.fc2=nn.Linear(30,50)
        self.fc3=nn.Linear(50,30)
        self.fc4=nn.Linear(30,nb_action)
    
    def forward(self,state):
        #using rectifier function to activate neurons
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        q_value=self.fc4(x)
        return q_value


#Implementing Experience Replay: Learning from Experience
#Long term memory for experience of last 100000 states
class ReplayMemory(object):
    
    def __init__(self,capacity=100000):
        self.capacity=capacity
        self.memory=[]
        
    def push(self,event):
        """event is tuple of 4 elements
            1st: last state=st
            2nd: new state= st+1
            3rd: last action= at
            4th:last reward= rd"""
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
    
    def sample(self,batch_size):
        #to get random samples from memory
        samples=zip(*random.sample(self.memory,batch_size)) #zip is to reshape function
        #we want format of 4 tuples of different elements of memory queue with each tuple size of batch_size
        return map(lambda x: Variable(torch.cat(x,0)),samples)
        

#Implementing Deep Q Leaning Model
class DQN:
    
    def __init__(self,input_size=5,nb_action=3,gamma=0.9):
        self.gamma=gamma
        self.reward_window=[]
        self.model=Network(input_size,nb_action)
        self.memory=ReplayMemory(100000)
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        self.last_action=0 #either 0,1,2
        self.last_reward=0
         
    def select_action(self,state):
        #state=input state/values of neural network
        probs=F.softmax(self.model(Variable(state))*100) #T=100
        action=probs.multinomial(1) #to randomly draw an action
        return action.data[0,0] #To get real action
    
    def learn(self,batch_state,batch_next_state,batch_reward,batch_action):
        outputs=self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) 
        #to get action chosen by network and destroy fake batch as we want the output as normal Tensor/vector
        next_outputs=self.model(batch_next_state).detach().max(1)[0]
        #detach: to get all posible outcomes of next state; out of which we select the best option
        target=self.gamma*next_outputs+batch_reward
        td_loss=F.smooth_l1_loss(outputs,target) #temporal difference=real Q-Q*
        self.optimizer.zero_grad()
        td_loss.backward() #back propagation
        self.optimizer.step() #weights getting updated
    
    def update(self,reward,new_signal):
        #Update everything when car attains new set
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([(self.last_reward)])))
        action=self.select_action(new_state)
        if len(self.memory.memory)>100:
            batch_state,batch_next_state, batch_action, batch_reward=self.memory.sample(100)
            self.learn(batch_state,batch_next_state, batch_reward, batch_action)
        self.last_action=action
        self.last_state=new_state
        self.last_reward=reward
        self.reward_window.append(reward)
        if len(self.reward_window)>1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return (sum(self.reward_window)/(len(self.reward_window)+1.))
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint=torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("DONE!!")
        else:
            print("NO CHECKPOINT FOUND!!")