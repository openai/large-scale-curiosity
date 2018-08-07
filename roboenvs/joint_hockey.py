# Make a basic version of pong, run it with random agent.

import os
import sys

import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import numpy as np
import roboschool
from roboschool.scene_abstract import Scene, cpp_household


class HockeyScene(Scene):
    # multiplayer = False
    # players_count = 1
    VIDEO_W = 84
    VIDEO_H = 84
    TIMEOUT = 300

    def __init__(self):
        Scene.__init__(self, gravity=9.8, timestep=0.0165 / 4, frame_skip=8)
        self.score_left = 0
        self.score_right = 0

    def actor_introduce(self, robot):
        i = robot.player_n - 1

    def episode_restart(self):
        Scene.episode_restart(self)
        if self.score_right + self.score_left > 0:
            sys.stdout.write("%i:%i " % (self.score_left, self.score_right))
            sys.stdout.flush()
        self.mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "roboschool_hockey.xml"))
        dump = 0
        for r in self.mjcf:
            if dump: print("ROBOT '%s'" % r.root_part.name)
            for part in r.parts:
                if dump: print("\tPART '%s'" % part.name)
                # if part.name==self.robot_name:
            for j in r.joints:
                if j.name == "p0x": self.p0x = j
                if j.name == "p0y": self.p0y = j
                if j.name == "p1x": self.p1x = j
                if j.name == "p1y": self.p1y = j
                if j.name == "ballx": self.ballx = j
                if j.name == "bally": self.bally = j
        self.ballx.set_motor_torque(0.0)
        self.bally.set_motor_torque(0.0)
        for r in self.mjcf:
            r.query_position()
        fpose = cpp_household.Pose()
        fpose.set_xyz(0, 0, -0.04)
        self.field = self.cpp_world.load_thingy(
            os.path.join(os.path.dirname(roboschool.__file__), "models_outdoor/stadium/pong1.obj"),
            fpose, 1.0, 0, 0xFFFFFF, True)
        self.camera = self.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        self.camera_itertia = 0
        self.frame = 0
        self.jstate_for_frame = -1
        self.score_left = 0
        self.score_right = 0
        self.bounce_n = 0
        self.restart_from_center(self.np_random.randint(2) == 0)

    def restart_from_center(self, leftwards):
        self.ballx.reset_current_position(0, 0)
        self.bally.reset_current_position(0, 0)
        self.timeout = self.TIMEOUT
        self.timeout_dir = (-1 if leftwards else +1)
        # self.bounce_n = 0
        self.ball_x, ball_vx = self.ballx.current_position()
        self.ball_y, ball_vy = self.bally.current_position()

    def global_step(self):
        self.frame += 1

        # if not self.multiplayer:
        #     # Trainer
        #     self.p1x.set_servo_target( self.trainer_x, 0.02, 0.02, 4 )
        #     self.p1y.set_servo_target( self.trainer_y, 0.02, 0.02, 4 )

        Scene.global_step(self)

        self.ball_x, ball_vx = self.ballx.current_position()
        self.ball_y, ball_vy = self.bally.current_position()

        if np.abs(self.ball_y) > 1.0 and self.ball_y * ball_vy > 0:
            self.bally.reset_current_position(self.ball_y, -ball_vy)

        if ball_vx * self.timeout_dir < 0:
            # if self.timeout_dir < 0:
            #     self.score_left += 0.00*np.abs(ball_vx)   # hint for early learning: hit the ball!
            # else:
            #     self.score_right += 0.00*np.abs(ball_vx)
            self.timeout_dir *= -1
            self.timeout = self.TIMEOUT
            self.bounce_n += 1

    def global_state(self):
        if self.frame == self.jstate_for_frame:
            return self.jstate
        self.jstate_for_frame = self.frame
        j = np.array(
            [j.current_relative_position() for j in [self.p0x, self.p0y, self.p1x, self.p1y, self.ballx, self.bally]]
        ).flatten()
        self.jstate = np.concatenate([j, [(self.timeout - self.TIMEOUT) / self.TIMEOUT]])
        return self.jstate

    def HUD(self, a, s):
        self.cpp_world.test_window_history_advance()
        self.cpp_world.test_window_observations(s.tolist())
        self.cpp_world.test_window_actions(a[:2].tolist())
        s = "%04i TIMEOUT%3i %0.2f:%0.2f" % (
            self.frame, self.timeout, self.score_left, self.score_right
        )

    def camera_adjust(self):
        "Looks like first 3 coordinates specify position of the camera and the last three the orientation."
        self.camera.move_and_look_at(0.1, -0.1, 1.9, 0.1, 0.2, 0)


class RoboschoolHockeyJoint(gym.Env):
    VIDEO_W = 84
    VIDEO_H = 84

    def __init__(self):
        self.player_n = 0
        self.scene = None
        action_dim = 4
        # obs_dim = 13
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.VIDEO_W, self.VIDEO_H, 3), dtype=np.uint8)
        self._seed()

    def create_single_player_scene(self):
        self.player_n = 0
        s = HockeyScene()
        s.np_random = self.np_random
        return s

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.scene is None:
            self.scene = self.create_single_player_scene()
        self.scene.episode_restart()
        s = self.calc_state()
        self.score_reported = 0
        obs = self.render("rgb_array")
        return obs

    def calc_state(self):
        j = self.scene.global_state()
        if self.player_n == 1:
            #                    [  0,1,  2,3,   4, 5, 6,7,  8,9,10,11,12]
            #                    [p0x,v,p0y,v, p1x,v,p1y,v, bx,v,by,v, T]
            signflip = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1])
            reorder = np.array([4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12])
            j = (j * signflip)[reorder]
        return j

    def apply_action(self, a0, a1):
        assert (np.isfinite(a0).all())
        assert (np.isfinite(a1).all())
        a0 = np.clip(a0, -1, +1)
        a1 = np.clip(a1, -1, +1)
        self.scene.p0x.set_target_speed(3 * float(a0[0]), 0.05, 7)
        self.scene.p0y.set_target_speed(3 * float(a0[1]), 0.05, 7)
        self.scene.p1x.set_target_speed(-3 * float(a1[0]), 0.05, 7)
        self.scene.p1y.set_target_speed(3 * float(a1[1]), 0.05, 7)

    def step(self, a):
        a0 = a[:2]
        a1 = a[2:]
        self.apply_action(a0, a1)
        self.scene.global_step()
        state = self.calc_state()
        self.scene.HUD(a, state)
        new_score = self.scene.bounce_n
        # new_score = int(new_score)
        self.rewards = new_score - self.score_reported
        self.score_reported = new_score
        if (self.scene.score_left > 10) or (self.scene.score_right > 10):
            done = True
        else:
            done = False
        obs = self.render("rgb_array")
        return obs, self.rewards, done, {}

    def render(self, mode):
        if mode == "human":
            return self.scene.cpp_world.test_window()
        elif mode == "rgb_array":
            self.scene.camera_adjust()
            rgb, _, _, _, _ = self.scene.camera.render(False, False,
                                                       False)  # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape((self.VIDEO_H, self.VIDEO_W, 3))
            return rendered_rgb
        else:
            assert (0)


class MultiDiscreteToUsual(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)

        self._inp_act_size = self.env.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self._inp_act_size))

    def action(self, a):
        vec = np.zeros(dtype=np.int8, shape=self._inp_act_size.shape)
        for i, n in enumerate(self._inp_act_size):
            vec[i] = a % n
            a /= n
        return vec


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env=None, nsamples=11):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self._dist_to_cont = []
        for low, high in zip(env.action_space.low, env.action_space.high):
            self._dist_to_cont.append(np.linspace(low, high, nsamples))
        temp = [nsamples for _ in self._dist_to_cont]
        self.action_space = gym.spaces.MultiDiscrete(temp)

    def action(self, action):
        assert len(action) == len(self._dist_to_cont)
        return np.array([m[a] for a, m in zip(action, self._dist_to_cont)], dtype=np.float32)
