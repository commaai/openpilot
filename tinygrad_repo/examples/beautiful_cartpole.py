from typing import Tuple
import time
from tinygrad import Tensor, TinyJit, nn
import gymnasium as gym
from tinygrad.helpers import trange
import numpy as np  # TODO: remove numpy import

ENVIRONMENT_NAME = 'CartPole-v1'
#ENVIRONMENT_NAME = 'LunarLander-v2'

#import examples.rl.lightupbutton
#ENVIRONMENT_NAME = 'PressTheLightUpButton-v0'

# *** hyperparameters ***
# https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md

BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 2000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 40
DISCOUNT_FACTOR = 0.99

class ActorCritic:
  def __init__(self, in_features, out_features, hidden_state=HIDDEN_UNITS):
    self.l1 = nn.Linear(in_features, hidden_state)
    self.l2 = nn.Linear(hidden_state, out_features)

    self.c1 = nn.Linear(in_features, hidden_state)
    self.c2 = nn.Linear(hidden_state, 1)

  def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
    x = self.l1(obs).tanh()
    act = self.l2(x).log_softmax()
    x = self.c1(obs).relu()
    return act, self.c2(x)

def evaluate(model:ActorCritic, test_env:gym.Env) -> float:
  (obs, _), terminated, truncated = test_env.reset(), False, False
  total_rew = 0.0
  while not terminated and not truncated:
    act = model(Tensor(obs))[0].argmax().item()
    obs, rew, terminated, truncated, _ = test_env.step(act)
    total_rew += float(rew)
  return total_rew

if __name__ == "__main__":
  env = gym.make(ENVIRONMENT_NAME)

  model = ActorCritic(env.observation_space.shape[0], int(env.action_space.n))    # type: ignore
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

  @TinyJit
  def train_step(x:Tensor, selected_action:Tensor, reward:Tensor, old_log_dist:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    with Tensor.train():
      log_dist, value = model(x)
      action_mask = (selected_action.reshape(-1, 1) == Tensor.arange(log_dist.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)).float()

      # get real advantage using the value function
      advantage = reward.reshape(-1, 1) - value
      masked_advantage = action_mask * advantage.detach()

      # PPO
      ratios = (log_dist - old_log_dist).exp()
      unclipped_ratio = masked_advantage * ratios
      clipped_ratio = masked_advantage * ratios.clip(1-PPO_EPSILON, 1+PPO_EPSILON)
      action_loss = -unclipped_ratio.minimum(clipped_ratio).sum(-1).mean()

      entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()   # this encourages diversity
      critic_loss = advantage.square().mean()
      opt.zero_grad()
      (action_loss + entropy_loss*ENTROPY_SCALE + critic_loss).backward()
      opt.step()
      return action_loss.realize(), entropy_loss.realize(), critic_loss.realize()

  @TinyJit
  def get_action(obs:Tensor) -> Tensor:
    ret = model(obs)[0].exp().multinomial().realize()
    return ret

  st, steps = time.perf_counter(), 0
  Xn, An, Rn = [], [], []
  for episode_number in (t:=trange(EPISODES)):
    get_action.reset()   # NOTE: if you don't reset the jit here it captures the wrong model on the first run through

    obs:np.ndarray = env.reset()[0]
    rews, terminated, truncated = [], False, False
    # NOTE: we don't want to early stop since then the rewards are wrong for the last episode
    while not terminated and not truncated:
      # pick actions
      # TODO: what's the temperature here?
      act = get_action(Tensor(obs)).item()

      # save this state action pair
      # TODO: don't use np.copy here on the CPU, what's the tinygrad way to do this and keep on device? need __setitem__ assignment
      Xn.append(np.copy(obs))
      An.append(act)

      obs, rew, terminated, truncated, _ = env.step(act)
      rews.append(float(rew))
    steps += len(rews)

    # reward to go
    # TODO: move this into tinygrad
    discounts = np.power(DISCOUNT_FACTOR, np.arange(len(rews)))
    Rn += [np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]

    Xn, An, Rn = Xn[-REPLAY_BUFFER_SIZE:], An[-REPLAY_BUFFER_SIZE:], Rn[-REPLAY_BUFFER_SIZE:]
    X, A, R = Tensor(Xn), Tensor(An), Tensor(Rn)

    # TODO: make this work
    #vsz = Variable("sz", 1, REPLAY_BUFFER_SIZE-1).bind(len(Xn))
    #X, A, R = Tensor(Xn).reshape(vsz, None), Tensor(An).reshape(vsz), Tensor(Rn).reshape(vsz)

    old_log_dist = model(X)[0].detach()   # TODO: could save these instead of recomputing
    for i in range(TRAIN_STEPS):
      samples = Tensor.randint(BATCH_SIZE, high=X.shape[0]).realize()  # TODO: remove the need for this
      # TODO: is this recompiling based on the shape?
      action_loss, entropy_loss, critic_loss = train_step(X[samples], A[samples], R[samples], old_log_dist[samples])
    t.set_description(f"sz: {len(Xn):5d} steps/s: {steps/(time.perf_counter()-st):7.2f} action_loss: {action_loss.item():7.3f} entropy_loss: {entropy_loss.item():7.3f} critic_loss: {critic_loss.item():8.3f} reward: {sum(rews):6.2f}")

  test_rew = evaluate(model, gym.make(ENVIRONMENT_NAME, render_mode='human'))
  print(f"test reward: {test_rew}")
