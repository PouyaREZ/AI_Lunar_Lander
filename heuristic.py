# import library
import sys, math
import numpy as np
import gym

def heuristic(env, s):
    '''Heuristic action for LunarLander-v2 using rule-based'''
    # get state from LunarLander-v2 environment
    p_x, p_y, v_x, v_y, alpha, omega, left, right = s
    
    # initialize the action as "0:Do Nothing"
    a = 0

    # if-else/condition-based action
    # centering the lander by "1:Fire Left Engine" or "3:Fire Right Engine" from absolute position from center
    if p_x > 0.1:
        a = 1
    elif p_x < -0.1:
        a = 3
    # stabilize the lander by "1:Fire Left Engine" or "3:Fire Right Engine" from abosolute angle from vertical
    if alpha > 0.1:
        a = 3
    elif alpha < -0.1:
        a = 1
    # stabilize the lander by "1:Fire Left Engine" or "3:Fire Right Engine" from abosolute angular velocity (positive and negative)
    if omega > 0.3:
        a = 3
    elif omega < -0.3:
        a = 1
    # reduce vertical velocity of the lander by "1:Fire Main Engine" 
    if v_y < -0.1:
        a = 2
    # stop all action when landed as "0:Do Nothing"
    if left == 1 or right == 1:
        a = 0
    
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    for _ in range(1000):
        env.seed(seed)
        total_reward = 0
        steps = 0
        s = env.reset()
        while True:
            a = heuristic(env, s)
            s, r, done, info = env.step(a)
            total_reward += r

            if render:
                still_open = env.render()
                if still_open == False: break

            if steps % 20 == 0 or done:
                print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1

            if done: 
                break
    return total_reward

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    demo_heuristic_lander(env, render=True)
