import tractor_trailer_envs as tt_envs
import pprint
import gymnasium as gym

from config import get_config
parser = get_config()
args = parser.parse_args()
config = {
            "vehicle_type": args.vehicle_type,
            "reward_type": args.reward_type,
            "xmax": args.xmax, 
            "ymax": args.ymax, # [m]
            "distancematrix": args.distance_weights,
            "reward_weights": args.reward_weights,
            "max_episode_steps": args.max_episode_steps,
            "goal": tuple(args.goal),
            "evaluate_mode": args.evaluate_mode,
            "allow_backward": args.allow_backward,
            "constraint_coeff": args.constraint_coeff,
            "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
            "sucess_goal_reward_others": args.sucess_goal_reward_others,
            "use_stable_baseline": args.use_stable_baseline,
            "verbose": args.verbose,
        }
env = tt_envs.TractorTrailerParkingEnv(config)
# env.seed(seed=40)
env.action_space.seed(seed=40)
for _ in range(3):
    obs , _ = env.reset()
    # state_list = [state]
    reward_list = []
    terminated = False
    truncated = False
    while (not terminated) and (not truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        # env.render()
        reward_list.append(reward) 
        # state_list.append(state)
print(1)
# env.run_simulation()
    