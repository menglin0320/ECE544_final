from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.lib import actions

import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_net(minimap, screen, info, msize, ssize, num_action, ntype, reuse = False):
  if ntype == 'atari':
    return build_atari(minimap, screen, info, msize, ssize, num_action, reuse = reuse)

  elif ntype == 'fcn':
    feat_fc,feat_conv = build_fcn(minimap, screen, info, msize, ssize, num_action,reuse = reuse)
    return build_a3c_part2(feat_fc,feat_conv)

  else:
    raise 'FLAGS.net must be atari or fcn'

def build_pc_net(pc_minimap, pc_screen, pc_info, msize, ssize, num_action,valid_non_spatial_action):
    feat_fc,feat_conv = build_fcn(pc_minimap, pc_screen, pc_info, msize, ssize, num_action,reuse = True)
    return build_pc_part2(feat_fc,feat_conv,valid_non_spatial_action)
  
def _fc_variable(self, weight_shape, name):
  name_w = "W_{0}".format(name)
  name_b = "b_{0}".format(name)
  
  input_channels  = weight_shape[0]
  output_channels = weight_shape[1]
  bias_shape = [output_channels]
  weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
  bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
  return weight, bias

def _conv_variable(self, weight_shape, name, deconv=False):
  name_w = "W_{0}".format(name)
  name_b = "b_{0}".format(name)
   
  w = weight_shape[0]
  h = weight_shape[1]
  if deconv:
    input_channels  = weight_shape[3]
    output_channels = weight_shape[2]
  else:
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
  bias_shape = [output_channels]
  weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
  bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
  return weight, bias

def build_atari(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value


def build_fcn(minimap, screen, info, msize, ssize, num_action, reuse = False):
  with tf.variable_scope("base_conv", reuse=reuse) as scope:
  # Extract features
    mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=5,
                           stride=1,
                           scope='mconv1')
    mconv2 = layers.conv2d(mconv1,
                           num_outputs=32,
                           kernel_size=3,
                           stride=1,
                           scope='mconv2')
    sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=5,
                           stride=1,
                           scope='sconv1')
    sconv2 = layers.conv2d(sconv1,
                           num_outputs=32,
                           kernel_size=3,
                           stride=1,
                           scope='sconv2')
    info_fc = layers.fully_connected(layers.flatten(info),
                                     num_outputs=256,
                                     activation_fn=tf.tanh,
                                     scope='info_fc')

    # Compute spatial actions
    feat_conv = tf.concat([mconv2, sconv2], axis=3)
    # Compute non spatial actions and value
    feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='feat_fc')
    return feat_fc, feat_conv  

def build_a3c_part2(feat_conv,feat_fc):
  spatial_action = layers.conv2d(feat_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value

def build_pc_part2(feat_fc,valid_non_spatial_action):
  with tf.variable_scope("pc_deconv", reuse=reuse) as scope:
    
    W_pc_fc1, b_pc_fc1 = self._fc_variable([256, 9*9*32], "pc_fc1")
          
    W_pc_deconv_v, b_pc_deconv_v = self._conv_variable([4, 4, 1, 32],
                                                           "pc_deconv_v", deconv=True)
    W_pc_deconv_a, b_pc_deconv_a = self._conv_variable([4, 4, len(actions.FUNCTIONS), 32],
                                                           "pc_deconv_a", deconv=True)
    h_pc_fc1 = tf.nn.relu(tf.matmul(feat_fc, W_pc_fc1) + b_pc_fc1)
    h_pc_fc1_reshaped = tf.reshape(h_pc_fc1, [-1,9,9,32])
    h_pc_deconv_v = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                  W_pc_deconv_v, 9, 9, 2) +
                                   b_pc_deconv_v)
    h_pc_deconv_a = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                  W_pc_deconv_a, 9, 9, 2) +
                                   b_pc_deconv_a)
    # Advantage mean
    h_pc_deconv_a_mean = tf.reduce_mean(h_pc_deconv_a, reduction_indices=3, keep_dims=True)
    pc_q = h_pc_deconv_v + h_pc_deconv_a - h_pc_deconv_a_mean
    valid_reshaped = tf.reshape(valid_non_spatial_action, [-1, 1, 1, len(actions.FUNCTIONS)])
    valid_pc_q = tf.multiply(valid_reshaped, pc_q)
    valid_pc_q = valid_pc_q/tf.reduce_sum(valid_pc_q,keep_dims = True, reduction_indices = 3)  
    pc_q_max = tf.reduce_max(valid_pc_q, reduction_indices=3, keep_dims=False)
    return valid_pc_q, pc_q_max