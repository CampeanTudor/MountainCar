import math

from gym import *
import gym
import numpy as np


class MountainCarEnvWrapper(gym.Wrapper):



    def step_with_hardcoded_values(self, position, velocity):

        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        self.state = [position, velocity]

        reward = -1.0

        return np.array(self.state), reward, done, {}

    def set_state_with_harcoded_values(self, position, velocity):

        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = np.clip(position, self.min_position, self.max_position)

        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        self.state = [position, velocity]

        reward = -1.0

        return np.array(self.state), reward, done, {}

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def step_with_custom_reward(self, action, new_action=True):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        if not done:
            reward = np.abs(position - self.previous_state[0]) - 0.025
        else:
            reward = 10

        if new_action:
            self.previous_state = [position, velocity]

        self.state = (position, velocity)

        return np.array(self.state), reward, done, {}

    def reset(self):

        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.previous_state = self.state

        return np.array(self.state)


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

