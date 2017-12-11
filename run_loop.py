from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils as U

import time
def calc_pixel_change(obs2, obs1):
  screen1 = np.array(obs1.observation['screen'], dtype=np.float32)
  screen1 = np.expand_dims(U.preprocess_screen(screen1), axis=0)
  screen2 = np.array(obs2.observation['screen'], dtype=np.float32)
  screen2 = np.expand_dims(U.preprocess_screen(screen2), axis=0)
  
  screen1_avg = np.mean(screen1,axis = 0)
  screen2_avg = np.mean(screen1,axis = 0)

  d = np.absolute(screen2_avg[2:-2,2:-2,:] - screen1_avg[2:-2,2:-2,:])
  # (60,60,3), any channel here? but with the np.mean on next line, doesn't matter either way.
  m = np.mean(d, 2)
  pc = self._subsample(m, 3)
  return pc

def _subsample(a, average_width):
  s = a.shape
  sh = s[0]//average_width, average_width, s[1]//average_width, average_width
  #20, 3, 20, 3
  return a.reshape(sh).mean(-1).mean(1) 

def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        pixel_change = calc_pixel_change(timesteps, last_timesteps)
        # Only for a single player!
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], pixel_change[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)

def random_run_loop( env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()
  try:
    while True:
      num_frames = 0
      #initialize timesteps
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [U.random_run(timesteps[0]),]
        timesteps = env.step(actions)
        pixel_change = calc_pixel_change(timesteps, last_timesteps)
        # Only for a single player!
        is_done = (num_frames >= max_frames)
        yield [last_timesteps[0], actions[0], pixel_change[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
