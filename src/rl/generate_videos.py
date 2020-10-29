import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from PIL import Image
import sys
from collections import namedtuple
import os.path
sys.path.insert(0, '/u/pgoyal/Research/metaworld')
from metaworld.envs.mujoco.sawyer_xyz.sawyer_random import SawyerRandomEnv
from ppo import Memory, ActorCritic, PPO
from utils import objects, obj2grp

DATASET_DIR = 'metaworld-dataset'

def create_random_env(obj_id):
    # randonly sample positions until collision
    positions = []
    while True:
        x_pos = np.random.randint(5)
        y_pos = np.random.randint(5)
        collision = False
        for (x, y) in positions:
            if abs(x-x_pos) < 2 and abs(y-y_pos) < 2:
                collision = True
                break
        else:
            positions.append((x_pos, y_pos))
        if collision:
            break

    print(positions)

    obj_ids = []
    for pos in positions:
        while True:
            if len(obj_ids) == 0:
                obj_id = obj_id
            else:
                obj_id = np.random.choice(range(len(objects)))
            obj = objects[obj_id]
            for obj_ in obj_ids:
                if obj2grp[obj] == obj2grp[objects[obj_]]:
                    break
            else:
                obj_ids.append(obj_id)
                print(pos, objects[obj_id])
                break

    return positions, obj_ids

def save_env(objects, positions, obj_ids, filename):
    with open(filename, 'w') as f:
        for i in range(len(positions)):
            f.write('{}\t{}\n'.format(positions[i], obj_ids[i]))

def main(args):
    ############## Hyperparameters ##############
    render = False
    solved_reward = 1e10         # stop training if avg_reward > solved_reward
    log_interval = 1           # print avg reward in the interval
    save_interval = 500
    max_episodes = 5000        # max training episodes
    max_timesteps = 500
    
    update_timestep = max_timesteps      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################


    trial = 0
    env_id = args.start
    save_dir = '{}/obj{}-env{}/'.format(DATASET_DIR, args.obj_id, env_id)
    while os.path.exists('{}/env.txt'.format(save_dir)):
        env_id += 1
        save_dir = '{}/obj{}-env{}/'.format(DATASET_DIR, args.obj_id, env_id)

    while env_id < args.end:
        trial += 1
        # creating environment
        positions, obj_ids = create_random_env(args.obj_id)
        state_dim = 6
        action_dim = 4

        memory = Memory()
        ppo = PPO(args, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
        print(lr,betas)
        
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0
        success_history = []
        
        env = SawyerRandomEnv(
            objects=objects, 
            positions=positions, 
            obj_ids=obj_ids, 
            state_rep='feature', 
            reward_type='dense',
            max_timesteps = max_timesteps)
        training_success = False

        # training loop
        for i_episode in range(1, max_episodes+1):
            state = env.reset()

            for t in range(max_timesteps):
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state, reward, done, success = env.step(action)
                # Saving reward:
                memory.rewards.append(reward)
                
                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0
                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            
            avg_length += t

            success_history.append(success)
                
            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))
                
                print('Trial: {} \t Episode: {} \t Length: {} \t Reward: {}'.format(trial, i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0

            # if last 5 episodes successful, save policy and generate video
            if sum(success_history[-5:]) == 5:
                training_success = True
                break

        if training_success:
            policy_filename = '{}/policy.pt'.format(save_dir)
            env_filename = '{}/env.txt'.format(save_dir)
            torch.save(ppo.policy.state_dict(), policy_filename)
            save_env(objects, positions, obj_ids, env_filename)

            # create and save video
            while True:
                frames_left_traj = []
                frames_center_traj = []
                frames_right_traj = []
                ee_pos_traj = []
                actions_traj = []

                state = env.reset()
                (frame_left, frame_center, frame_right, ee) = env.get_frame()
                frames_left_traj.append(frame_left)
                frames_center_traj.append(frame_center)
                frames_right_traj.append(frame_right)
                ee_pos_traj.append(ee)
                video_done = False
                for t in range(max_timesteps):
                    time_step += 1
                    action = ppo.select_action(state, memory)
                    state, reward, done, success = env.step(action)
                    actions_traj.append(action)
                    (frame_left, frame_center, frame_right, ee) = env.get_frame()
                    frames_left_traj.append(frame_left)
                    frames_center_traj.append(frame_center)
                    frames_right_traj.append(frame_right)
                    ee_pos_traj.append(ee)
                    if done:
                        video_done = success
                        break

                if video_done:
                    # save frames, actions, ee_pos
                    for idx, frame in enumerate(frames_left_traj):
                        img = Image.fromarray(frame)
                        img.save('{}/left-{}.png'.format(save_dir, idx))
                    for idx, frame in enumerate(frames_center_traj):
                        img = Image.fromarray(frame)
                        img.save('{}/center-{}.png'.format(save_dir, idx))
                    for idx, frame in enumerate(frames_right_traj):
                        img = Image.fromarray(frame)
                        img.save('{}/right-{}.png'.format(save_dir, idx))
                    np.save('{}/actions.npy'.format(save_dir), actions_traj)
                    np.save('{}/ee_pos.npy'.format(save_dir), ee_pos_traj)
                    break
                        
            env_id += 1
            save_dir = '{}/obj{}-env{}/'.format(DATASET_DIR, args.obj_id, env_id)
            while os.path.exists('{}/env.txt'.format(save_dir)):
                env_id += 1
                save_dir = '{}/obj{}-env{}/'.format(DATASET_DIR, args.obj_id, env_id)

def get_args():
    import argparse
    parser = argparse.ArgumentParser('Train PPO policy')
    parser.add_argument('--obj-id', type=int, help='Index of main object; 0-12')
    parser.add_argument('--start', type=int, help='')
    parser.add_argument('--end', type=int, help='')
    args = parser.parse_args()
    return args
            
if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
    
