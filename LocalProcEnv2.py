# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from collections import deque
import random
import cv2
import torch
import procgen
import numpy as np 

class Env:

  def __init__(self, args):
    
    self.env= procgen.ProcgenEnv(1,args.game ,distribution_mode='easy')
    self.training = True  # Consistent with model training mode
    self.actions=range(self.env.action_space.n)
    self.device = args.device
    self.window= args.history_length
    self.state_buffer = deque([], maxlen=args.history_length)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      ob, rew, done,_ = self.env.step(np.array([action]))
      reward += rew
      if t == 2:
        frame_buffer[0] = self._get_state(ob)
      elif t == 3:
        frame_buffer[1] = self._get_state(ob)
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  def rgb2gray(self, rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84 ,dtype=torch.float32, device=self.device))

  def _get_state(self,ob):
    state = self.rgb2gray(ob['rgb'][0])
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
    
  def reset(self):
    # Reset internals
    self._reset_buffer()
    observation = self._get_state(self.env.reset())
    # Process and return "initial" state
    self.state_buffer.append(observation)
    return torch.stack(list(self.state_buffer), 0)
  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
