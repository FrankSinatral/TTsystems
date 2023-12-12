import time
import joblib
import os
import os.path as osp
# import tensorflow as tf
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../TTsystems_and_PINN/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../TTsystems_and_PINN/highway_embed")
# import rl_training.tt_system_implement as tt_envs
import rl_training.envs.tt_envs as tt_envs
from rl_training.config import get_config
# import HybridAstarPlanner.planner_base.no_obs_version as alg_no_obs
import VehicleModel.obstacle as obs
import numpy as np
from logx import EpochLogger
from logx import restore_tf_graph
import imageio
import re


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][6:]) for x in os.listdir(pytsave_path) if len(x)>9 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'pyt_vars', 'vars_'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model_'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            action = model.act(x)
        return action

    return get_action

def calculate_planner(env, o):
    ox, oy = obs.map_plain4_high_resolution()
    ox, oy = obs.remove_duplicates(ox, oy)
    goal = np.array([0, 0, 0])
    single_tractor_planner = alg_no_obs.SingleTractorHybridAstarPlanner(ox, oy)
    path, control_list, rs_path = single_tractor_planner.plan(o, goal, get_control_sequence=True)
    episode_return = 0.0
    state = o
    episode_length = 1
    # reward_list = []
    # state_list = [state]
    for control in control_list:
            state_ = single_tractor_planner.step(state, control)
            reward = env.parking_reward(state)
            # reward_list.append(reward)
            # state_list.append(state_)
            episode_return += reward
            episode_length += 1
            if reward >= -0.6:
                break
            state = state_
            
    return episode_return, episode_length

def save_gif(frames, dir="rl_training/gif/standard_parking", duration=50):
    if not os.path.exists(dir):
        os.makedirs(dir)
    base_path = os.path.join(dir, "path_simulation")
    extension = ".gif"
    
    all_files = os.listdir(dir)
    matched_files = [re.match(r'path_simulation(\d+)\.gif', f) for f in all_files]
    numbers = [int(match.group(1)) for match in matched_files if match]
    
    if numbers:
        save_index = max(numbers) + 1
    else:
        save_index = 1
    imageio.mimsave(base_path + str(save_index) + extension, frames, duration=duration)
    
def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=False, env_name="single_tractor",
               logger_kwargs=dict()):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    if env_name == "standard_parking":
        # here you can implement a visualize tool for parking
        
        # Also here you need to modify your saving place
        logger = EpochLogger(**logger_kwargs)
        seed = 0
        o, _ = env.reset(seed=seed)
        r, terminated, truncated, ep_ret, ep_len, n = 0, False, False, 0, 0, 0
        frames = []
        
        while n < num_episodes:
            
            if render:
                frame = env.render()
                frames.append(frame)
                time.sleep(1e-3)
            real_observation = (np.concatenate([o['observation'], o['desired_goal']])).astype(np.float32)
            a = get_action(real_observation)
            o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            if terminated or truncated:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                save_gif(frames)
                seed += 1
                o, _ = env.reset(seed=seed)
                r, terminated, truncated, ep_ret, ep_len = 0, False, False, 0, 0
                frames = []
                n += 1
                
    else:
    
        # here you can implement a visualize tool for parking
        
        # Also here you need to modify your saving place
        logger = EpochLogger()
        o, r, terminated, truncated, ep_ret, ep_len, n = env.reset(), 0, False, False, 0, 0, 0
        # here I add rs path as a refenrence (only for single tractor)
        # planner_episode_return, planner_episode_length = calculate_planner(env, o)
        # state_list = [o]
        # action_list = []
        # reward_list = []
        while n < num_episodes:
            
            if render:
                env.render()
                time.sleep(1e-3)
            o_extend = np.pad(o, (0, 6 - o.shape[0]), 'constant')
            a = get_action(o_extend)
            o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            # state_list.append(o)
            # action_list.append(a)
            # reward_list.append(r)

            if terminated or truncated:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                # print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, planner_episode_return, planner_episode_length))
                env.run_simulation()
                o, r, terminated, truncated, ep_ret, ep_len = env.reset(), 0, False, False, 0, 0
                # planner_episode_return, planner_episode_length = calculate_planner(env, o)
                # state_list = [o]
                # reward_list = []
                # action_list = []
                n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    env_name = args.env_name
    seed = args.seed
    exp_name = env_name + '_' + args.algo_name + '_' + str(seed)
    logger_kwargs = {
        'output_dir': args.evaluate_dir + exp_name,
        'output_fname': args.output_fname,
        'exp_name': exp_name,
     }
    fpath = args.logging_dir + exp_name 
    
    env, get_action = load_policy_and_env(fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    # rewrite the env here
    
    if env_name == "standard_parking":
        from highway_env import register_highway_envs
        register_highway_envs()
        import gymnasium as gym
        env = gym.make('parking-v0', render_mode='rgb_array')
        run_policy(env, 
                   get_action, 
                   args.len, 
                   args.episodes, 
                   render=True, 
                   env_name=args.env_name,
                   logger_kwargs=logger_kwargs)
    else:
        config = {
            "env_name": args.env_name,
            "reward_type": args.reward_type,
            "xmax": args.xmax, 
            "ymax": args.ymax, # [m]
            "distancematrix": args.distance_weights,
            "reward_weights": args.reward_weights,
            "max_episode_steps": args.max_episode_steps,
            "goal": tuple(args.goal),
            "penalty_backward": args.penalty_backward,
            "penalty_switch": args.penalty_switch,
            "edge": args.edge,
            "save_gif": args.save_gif,
            "allow_backward": args.allow_backward,
            "constraint_coeff": args.constraint_coeff,
            "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
            "sucess_goal_reward_others": args.sucess_goal_reward_others,
            "continuous_step": args.continuous_step,
        }
        env = tt_envs.TractorTrailerEnv(config, args)
        run_policy(env, 
                   get_action, 
                   args.len, 
                   args.episodes, 
                   not(args.norender), 
                   env_name=args.env_name,
                   logger_kwargs=logger_kwargs)
    
    print("Done Evaluation")