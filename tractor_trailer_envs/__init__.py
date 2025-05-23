from gymnasium.envs.registration import register
from tractor_trailer_envs.envs.parking_env import (
    TractorTrailerParkingEnv, 
    TractorTrailerParkingEnvVersion1
)
from tractor_trailer_envs.map_and_obstacles.settings import (
    remove_duplicates,
    is_convex_quadrilateral
)
from tractor_trailer_envs.map_and_obstacles.settings import (
    QuadrilateralObstacle,
    MapBound,
    EllipticalObstacle 
)
from tractor_trailer_envs.vehicles.vehicle_zoo import (
    SingleTractor,
    OneTrailer,
    TwoTrailer,
    ThreeTrailer
)

def register_tt_envs():
    register(
        id='tt-parking-v0',
        entry_point='tractor_trailer_envs.envs:TractorTrailerParkingEnv'
    )
    register(
        id="tt-parking-v1",
        entry_point='tractor_trailer_envs.envs:TractorTrailerParkingEnvVersion1'
    )
    register(
        id="tt-reaching-v0",
        entry_point='tractor_trailer_envs.envs:TractorTrailerReachingEnv'
    )
    register(
        id="tt-cluttered-reaching-v0",
        entry_point='tractor_trailer_envs.envs:TractorTrailerClutteredReachingEnv'
    )
    register(
        id="tt-meta-reaching-v0",
        entry_point='tractor_trailer_envs.envs:TractorTrailerMetaReachingEnv'
    )
    register(
        id="tt-planning-v0",
        entry_point='tractor_trailer_envs.envs:TractorTrailerMetaPlanningEnv'
    )
    # register(
    #     id="tt-reaching-v1",
    #     entry_point='tractor_trailer_envs.envs:TractorTrailerReachingEnvVersion1'
    # )