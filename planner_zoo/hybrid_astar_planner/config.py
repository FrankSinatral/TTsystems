import argparse

def get_config():
    
    parser = argparse.ArgumentParser(
        description="hybrid_astar_planning_tt", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ## ------------------------------ basic setting ----------------------------------- ##
    parser.add_argument(
        "--algo_name",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="the agent that you choose to train"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="three_tractor_trailer",
        choices=["single_tractor","three_tractor_trailer"],
        help="which env you are using"
    )
    parser.add_argument(
        "--seed", type=int, default=40, help="random seed for numpy/torch"
    )
    ## ------------------------------ env setting --------------------------------- ##
    parser.add_argument(
        "--reward_type",
        type=str,
        # it seems that diff_distance is the best now
        default="diff_distance",
        choices=["conditional", "smooth", "rsp", "diff_distance"],
        help="the type of reward used")
    
    
    parser.add_argument(
        "--whether_obs",
        action='store_true',
        help="Whether there's obstacle in your env"
    )
    parser.add_argument(
        "--heuristic_type",
        type=str,
        default="traditional",
        choices=["traditional", "critic"],
        help="which heuristic you choose to use in hybrid astar planning"
    )
    return parser
    