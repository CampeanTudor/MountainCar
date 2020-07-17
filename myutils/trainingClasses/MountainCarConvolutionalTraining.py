import csv
from collections import deque
import random

import keras
import cv2
from keras import models, Model
from keras import layers
from keras.optimizers import Adam
from myutils.offlineLearningDataGeneration.TrainingSetManipulator import TrainingSetManipulator
import myutils.constants.Constants as cts
from keras.losses import huber_loss
import numpy as np
from keras.utils import to_categorical


class MountainCarConvolutionalTraining:

    def __init__(self, env):

        self.env = env

        self.stack_depth, self.image_height, self.image_width = self.get_model_input_shape()
        self.num_actions = env.action_space.n

        self.gamma = 0.99

        self.epsilon = 1
        self.epsilon_decay = 0.000018
        self.epsilon_min = 0.1

        self.frames_memory = deque(maxlen=self.stack_depth)
        self.replay_buffer = deque(maxlen=100000)
        self.minimum_samples_for_training = 50000
        self.num_pick_from_buffer = 32

        self.time_steps_in_episode = 300  # max is 200

        self.episode_num = 10000

        self.training = False

        self.update_weights_threshold = 35
        self.save_model_threshold = 1000

        self.learning_rate = 0.00025
        self.train_network = self.create_network()
        self.target_network = self.create_network()

        self.training_set_manipulator = TrainingSetManipulator()

    def create_network(self):

        input_shape = (self.stack_depth, self.image_height, self.image_width)

        action_mask = layers.Input(shape=(self.num_actions,), name='action_mask')

        state_input = layers.Input(input_shape, name="state_input")
        conv_1 = layers.Conv2D(32, (8, 8), strides=4, padding="same", activation='relu', name='conv_1')(state_input)
        conv_2 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv_2')(conv_1)
        conv_3 = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv_3')(conv_2)

        flatten = layers.Flatten()(conv_3)

        dense_hidden = layers.Dense(512, activation='relu', name='dense_hidden')(flatten)
        output_Q_values = layers.Dense(self.num_actions, activation='linear', name='output_Q_values')(dense_hidden)

        output_Q_values_with_action_mask = layers.Multiply(name='output_Q_values_with_action_mask')([output_Q_values, action_mask])

        model = Model(input=[state_input, action_mask], output=[output_Q_values_with_action_mask])

        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))

        return model

    def start(self):

        for episode in range(self.episode_num):
            self.frames_memory.clear()

            self.env.reset()

            current_image = self.env.render(mode='rgb_array')
            current_frame = self.process_image(
                current_image)  # the frame is an greyscale image of the current position
            current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])
            current_state = np.repeat(current_frame, self.stack_depth, axis=0)
            self.frames_memory.extend(current_state)

            self.do_learn(current_state, episode)


    def do_learn(self, current_state, episode):

        reward_sum = 0

        for time_step in range(self.time_steps_in_episode):

            # calculate a new action only when all frames from a state had been changed
            if time_step % self.stack_depth == 0:
                best_action = self.get_best_action(current_state)

            new_state_numerical, reward, done, _ = self.env.step_with_custom_reward(best_action)
            new_image = self.env.render(mode='rgb_array')
            next_frame = self.process_image(new_image)
            next_frame = next_frame.reshape(next_frame.shape[0], next_frame.shape[1])

            # current_state is a FIFO buffer so just by appending the size  of current_state is constant
            self.frames_memory.append(next_frame)

            new_state = np.asarray(self.frames_memory)

            self.replay_buffer.append([current_state, best_action, reward, new_state, done])

            # make the training possible only when the minimum experience was gathered
            if len(self.replay_buffer) == self.minimum_samples_for_training:
                self.training = True


            if self.training:
                self.train_training_network()
            reward_sum += reward
            current_state = new_state

            if done:
                break

        if time_step >= self.time_steps_in_episode - 1:
            print("Failed to finish task in episode {} with reward {} and epsilon {}".format(episode, reward_sum,
                                                                                             self.epsilon))
        else:
            print("Success in epsoide {}, used {} time steps!".format(episode, time_step))
            self.train_network.save(
                cts.Constants.PATH_TO_MODELS_SUCCESSFUL + 'successful_while_training_episode_{}_timesteps_{}_rewards_{}.h5'.format(
                    episode, time_step, reward_sum))

        if self.training:

            # synchronize model_network and target_network
            if (episode % self.update_weights_threshold) == 0:
                self.synch_networks()

            # save weights for tracking progress
            if (episode % self.save_model_threshold) == 0:
                print('Data saved at episode:', episode)
                self.train_network.save(
                    cts.Constants.PATH_TO_MODELS_TRACKING_PROGRESS_TRESHOLD_SAVES + 'DQN_CNN_model_episode_{}.h5'.format(
                        episode, episode))
            with open('./rewards_in_episodes.csv', mode='a+', newline='') as numerical_data:
                numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                numerical_data_writer.writerow([episode, reward_sum])

    def get_best_action(self, state):
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
            action = np.argmax(self.train_network.predict([state, np.ones((1, self.num_actions))]))

        # update epsilon
        if self.training:
            self.epsilon -= self.epsilon_decay

        return action

    def train_training_network(self):

        samples = self.get_samples_batch()

        if not samples:
            return

        samples = np.asarray(samples)
        current_states = samples[:, 0]
        actions = samples[:, 1]
        rewards = samples[:, 2]
        new_states = samples[:, 3]
        dones = samples[:, 4]

        current_states = self.normalize_images(list(current_states))

        new_states = self.normalize_images(list(new_states))

        actions = np.asarray(list(actions))
        
        dones = list(dones)

        rewards = np.asarray(rewards)

        #the values of the next state the agent arrives used to update it's Q_value for the (current_state, action) from the sample
        next_state_Q_values = self.target_network.predict([new_states, np.repeat(np.ones((1, self.num_actions)), self.num_pick_from_buffer,axis=0)])

        #create the targets for the case when the final state is not terminal
        updated_Q_values = rewards + self.gamma * next_state_Q_values.max(axis=1)

        #if the final state is terminal than the pre-terminal state(current state) has Q = reward only
        updated_Q_values[dones] = rewards[dones]

        encoded_actions = to_categorical(actions, num_classes=3)

        updated_Q_values = encoded_actions * updated_Q_values[:, None]

        self.train_network.fit([current_states, encoded_actions], updated_Q_values, epochs=1, verbose=0)

    def get_samples_batch(self):

        if len(self.replay_buffer) < self.num_pick_from_buffer:
            return

        samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)

        return samples

    def get_model_input_shape(self):
        self.env.reset()
        initial_image_shape = self.env.render(mode='rgb_array').shape
        image_height = 48  # initial_image_shape[0]
        image_width = 48  # initial_image_shape[1]
        stack_depth = 4

        # dimensions are 2 400 600
        return stack_depth, image_height, image_width

    def synch_networks(self):
        self.target_network.set_weights(self.train_network.get_weights())

    def process_image(self, image):
        # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.resize(image, (48, 48))

    def normalize_images(self, image):
        return np.float32(np.true_divide(image, 255))
