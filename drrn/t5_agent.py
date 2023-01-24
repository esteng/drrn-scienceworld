import pickle
import pdb
import re 
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
from memory import ReplayMemory, PrioritizedReplayMemory, Transition, State, BigState
from t5_model import T5Model
from util import *
import logger
from drrn import DRRNAgent

import shutil
import os
import sys
import time

import signal
from contextlib import contextmanager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#   Timeout additions (from https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/ )
#
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    print("Timeout")
    raise TimeoutError

#
#   T5 agent
#
class T5Agent(DRRNAgent):
    def __init__(self, args):
        super(T5Agent, self).__init__(args)

        # overwrite 
        self.network = T5Model(args.model_name_or_path).to(device)

        ## TODO: optimize only part of network based on args 
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)


    def build_state(self, obs, infos, prev_actions, prev_observations):
        """ Build a state from the observation and infos """
        obs = [x.split("OBSERVATION")[1] for x in obs]
        task_desc = [x['taskDesc'] for x in infos]
        task_desc = [re.sub("Task Description:\s+", "", x) for x in task_desc]
        inv = [x['inv'] for x in infos]
        look = [x['look'] for x in infos]
        separated_prev_actions = [pa if pa is not None else "" for pa in prev_actions ]
        separated_prev_obs = [po if po is not None else "" for po in prev_observations]
        states = [BigState(o, td, i, l, pa, po) 
                    for o, td, i, l, pa, po in zip(obs, task_desc, inv, look, separated_prev_actions, separated_prev_obs)]

        return states

    def encode(self, obs_list):
        """ only used to encode actions, not needed here
        Can be handled in forward function of network"""
        return obs_list 


    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample)
        pdb.set_trace()
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1-torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()

        # TODO: clip only part of network based on args
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()


