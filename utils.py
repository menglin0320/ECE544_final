from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    else:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)

def _subsample(self, a, average_width):
  s = a.shape
  sh = s[0]//average_width, average_width, s[1]//average_width, average_width
  return a.reshape(sh).mean(-1).mean(1)

def _calc_pixel_change(self, state, last_state):
  d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
  # (80,80,3)
  m = np.mean(d, 2)
  c = self._subsample(m, 4)
  return c

def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      c += 1
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def screen_channel():
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      c += 1
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c

def random_run():
  act_id = np.random.choice(valid_actions)

  dy = np.random.randint(-4, 5)
  target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
  dx = np.random.randint(-4, 5)
  target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))
  act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
  return actions.FunctionCall(act_id, act_args)