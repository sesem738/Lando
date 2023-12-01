import gym
import time
import phoenix_drone_simulation

env = gym.make('DroneCircleBulletEnv-v0')

while True:
    done = False
    env.render()  # make GUI of PyBullet appear
    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
        time.sleep(1/60)