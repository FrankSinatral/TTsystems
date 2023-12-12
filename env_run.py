import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint



# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

import rl_agents as rl
from rl_agents.config import get_config
    
def gym_env_fn():
    import gymnasium as gym
    from highway_env import register_highway_envs
    register_highway_envs()
    return gym.make("parking-v0", render_mode="rgb_array")

def gym_tt_env_fn(config):
    import gymnasium as gym
    from tractor_trailer_envs import register_tt_envs
    register_tt_envs()
    return gym.make('tt-parking-v0', config=config)


def main():
    parser = get_config()
    args = parser.parse_args()
    env_name = args.env_name
    seed = args.seed
    exp_name = env_name + '_' + args.algo_name + '_' + str(seed) 
    logger_kwargs = {
        'output_dir': args.logging_dir + exp_name,
        'output_fname': args.output_fname,
        'exp_name': exp_name,
     }
    if args.activation == 'ReLU':
        # sac using ReLU
        activation_fn = nn.ReLU
    elif args.activation == 'Tanh':
        # ppo using Tanh
        activation_fn = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation function: {args.activation}")
    ac_kwargs = {
        "hidden_sizes": tuple(args.hidden_sizes),
        "activation": activation_fn
    }
    
    if args.env_name == "standard_parking":
        agent = rl.SAC(env_fn=gym_env_fn,
                    ac_kwargs=ac_kwargs,
                    seed=seed,
                    steps_per_epoch=args.sac_steps_per_epoch,
                    epochs=args.sac_epochs,
                    replay_size=args.replay_size,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    lr=args.lr,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    start_steps=args.start_steps,
                    update_after=args.update_after,
                    update_every=args.update_every,
                    # missing max_ep_len
                    logger_kwargs=logger_kwargs, 
                    save_freq=args.save_freq,
                    num_test_episodes=args.num_test_episodes,
                    log_dir=args.log_dir,
                    whether_her=args.whether_her,
                    use_automatic_entropy_tuning=args.use_auto,
                    args=args)
        agent.run()
        
    else:
        #TODO: our own tractor trailer env
        # config = {
        #     "env_name": args.env_name,
        #     "reward_type": args.reward_type,
        #     "xmax": args.xmax, 
        #     "ymax": args.ymax, # [m]
        #     "distancematrix": args.distance_weights,
        #     "reward_weights": args.reward_weights,
        #     "max_episode_steps": args.max_episode_steps,
        #     "goal": tuple(args.goal),
        #     "penalty_backward": args.penalty_backward,
        #     "penalty_switch": args.penalty_switch,
        #     "edge": args.edge,
        #     "save_gif": args.save_gif,
        #     "allow_backward": args.allow_backward,
        #     "constraint_coeff": args.constraint_coeff,
        #     "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
        #     "sucess_goal_reward_others": args.sucess_goal_reward_others,
        #     "continuous_step": args.continuous_step,
        # }
        config = {
            "vehicle_type": args.env_name,
            "reward_type": args.reward_type,
            # "distancematrix": args.distance_weights,
            # "reward_weights": args.reward_weights,
            "max_episode_steps": args.max_episode_steps,
            # "goal": tuple(args.goal),
            "evaluate_mode": args.evaluate_mode,
            "allow_backward": args.allow_backward,
            "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
            "sucess_goal_reward_others": args.sucess_goal_reward_others,
            "verbose": args.verbose,
            "outer_wall_bound": {
                "x_min": -50, #[m]
                "x_max": 50,
                "y_min": -50,
                "y_max": 50,
            },
            "start_region_bound": {
                "x_min": -10, #[m]
                "x_max": 10,
                "y_min": -10,
                "y_max": 10,
            }
        }
        
        agent = rl.SAC(env_fn=gym_tt_env_fn,
                    ac_kwargs=ac_kwargs,
                    seed=seed,
                    steps_per_epoch=args.sac_steps_per_epoch,
                    epochs=args.sac_epochs,
                    replay_size=args.replay_size,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    lr=args.lr,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    start_steps=args.start_steps,
                    update_after=args.update_after,
                    update_every=args.update_every,
                    # missing max_ep_len
                    logger_kwargs=logger_kwargs, 
                    save_freq=args.save_freq,
                    num_test_episodes=args.num_test_episodes,
                    log_dir=args.log_dir,
                    whether_her=args.whether_her,
                    use_automatic_entropy_tuning=args.use_auto,
                    config=config,
                    args=args)
        agent.run()
        
    print(1)

if __name__ == "__main__":
    main()
    
