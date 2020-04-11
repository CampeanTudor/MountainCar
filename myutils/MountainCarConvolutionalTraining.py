from collections import deque
import random

import cv2
from keras import models
from keras import layers
from keras.optimizers import Adam

import numpy as np


class MountainCarConvolutionalTraining:

    def __init__(self, env):

        self.env = env

        self.stack_depth, self.image_height, self.image_width = self.get_model_input_shape()
        self.num_actions = env.action_space.n

        self.learning_rate = 0.00025
        self.train_network = self.create_network()
        self.target_network = self.create_network()

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.1

        self.frames_memory = deque(maxlen=self.stack_depth)
        self.replay_buffer = deque(maxlen=20000)
        self.num_pick_from_buffer = 32

        self.iteration_num = 201  # max is 200

        self.episode_num = 400

    def get_model_input_shape(self):
        self.env.reset()
        initial_image_shape = self.env.render(mode='rgb_array').shape
        image_height = initial_image_shape[0]
        image_width = initial_image_shape[1]
        stack_depth = 2

        # dimensions are 2 400 600
        return stack_depth, image_height, image_width

    def create_network(self):
        input_shape = (self.stack_depth, self.image_height, self.image_width)

        model = models.Sequential()

        model.add(layers.Conv2D(32, (8, 8), strides=4, padding="same", activation='relu', input_shape=input_shape,
                                name='conv_1'))
        model.add(layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv_2'))
        model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv_3'))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu', name='dense_1'))
        model.add(layers.Dense(self.num_actions, activation='linear', name='output'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def get_best_action(self, state):

        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.train_network.predict(state)[0])

        return action

    def train_from_buffer(self):

        if len(self.replay_buffer) < self.num_pick_from_buffer:
            return

        samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)

        states = []
        new_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        states = np.array(states)
        new_states = np.array(new_states)

        targets = self.train_network.predict(states)
        new_state_targets = self.target_network.predict(new_states)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * 0.99
            i += 1

        self.train_network.fit(states, targets, epochs=1, verbose=0)

    def original_try(self, current_state, eps):

        reward_sum = 0

        for i in range(self.iteration_num):

            best_action = self.get_best_action(current_state)

            new_state_numerical, reward, done, _ = self.env.step(best_action)
            new_image = self.env.render(mode='rgb_array')
            next_frame = self.process_image(new_image)
            next_frame = next_frame.reshape(next_frame.shape[0], next_frame.shape[1])

            self.frames_memory.append(
                next_frame)  # current_state is a FIFO buffer so just by appending the size  of current_state is kept constant

            new_state = np.asarray(self.frames_memory)

            # # Adjust reward for task completion
            if done:
                reward += 10

            self.replay_buffer.append([current_state, best_action, reward, new_state, done])

            self.train_from_buffer()

            reward_sum += reward
            current_state = new_state

            # print(i)

            if done:
                break

        if i >= 199:
            print("Failed to finish task in epsoide {}".format(eps))
        else:
            print("Success in epsoide {}, used {} iterations!".format(eps, i))
            self.train_network.save(
                '../models/modelsMountainCar/trainNetworkInEPS{}with{}iterations.h5'.format(eps, i))

            # SYNC

            self.target_network.set_weights(self.train_network.get_weights())

            print("now epsilon is {}, the reward is {}".format(max(self.epsilon_min, self.epsilon), reward_sum))
            self.epsilon -= self.epsilon_decay

        print("finished episode in {}".format(i))

    def process_image(self, image):
        # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.resize(image, (self.image_width, self.image_height))

    def start(self):

        for eps in range(self.episode_num):
            self.frames_memory.clear()

            self.env.reset()

            current_image = self.env.render(mode='rgb_array')
            current_frame = self.process_image(current_image)  # the frame is an greyscale image of the current position
            current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])
            current_state = np.repeat(current_frame, self.stack_depth, axis=0)
            self.frames_memory.extend(current_state)

            self.original_try(current_state, eps)
