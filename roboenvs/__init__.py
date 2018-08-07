import gym
from gym.envs.registration import register

from .joint_pong import DiscretizeActionWrapper, MultiDiscreteToUsual

register(
    id='RoboschoolPong-v2',
    entry_point='.joint_pong:RoboschoolPongJoint',
    max_episode_steps=10000,
    tags={"pg_complexity": 20 * 1000000},
)

register(
    id='RoboschoolHockey-v1',
    entry_point='.joint_hockey:RoboschoolHockeyJoint',
    max_episode_steps=1000,
    tags={"pg_complexity": 20 * 1000000},
)


def make_robopong():
    return gym.make("RoboschoolPong-v2")


def make_robohockey():
    return gym.make("RoboschoolHockey-v1")
