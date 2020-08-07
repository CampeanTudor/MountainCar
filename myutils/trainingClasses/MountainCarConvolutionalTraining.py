import csv
import datetime
from collections import deque
import random

from keras.callbacks import History

import tensorflow as tf
import cv2
from tensorflow.keras import models, Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import myutils.constants.Constants as cts
import numpy as np
from keras.utils import to_categorical

from myutils.ModelTrainingAndValidationParameters import ModelTrainingAndValidationParameters


class MountainCarConvolutionalTraining:

    def __init__(self):

        self.config = ModelTrainingAndValidationParameters()

        self.env = self.config.ENVIRONMENT

        self.gamma = 0.99

        self.epsilon = 1
        self.epsilon_decay = 0.0004
        self.epsilon_min = 0.1

        self.learning_rate = 0.00025

        self.frames_memory = deque(maxlen=self.config.FRAMES_IN_STATE)
        self.replay_buffer = deque(maxlen=self.config.MAXIMUM_NUMBER_OF_STATES_IN_BUFFER)
        self.minimum_samples_for_training = self.config.MINIMUM_SAMPLES_TO_START_TRAINING
        self.num_pick_from_buffer = 32

        self.training = False

        self.train_network = self.create_network()
        self.target_network = self.create_network()

        self.training_model_history = History()

    def create_network(self):

        input_shape = (self.config.FRAMES_IN_STATE, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)

        action_mask = layers.Input(shape=(self.config.NUMBER_OF_ACTIONS,), name='action_mask')

        state_input = layers.Input(input_shape, name="state_input")
        conv_1 = layers.Conv2D(32, (8, 8), strides=4, padding="same", activation='relu', name='conv_1')(state_input)
        conv_2 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv_2')(conv_1)
        conv_3 = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv_3')(conv_2)

        flatten = layers.Flatten()(conv_3)

        dense_hidden = layers.Dense(512, activation='relu', name='dense_hidden')(flatten)
        output_Q_values = layers.Dense(self.config.NUMBER_OF_ACTIONS, activation='linear', name='output_Q_values')(dense_hidden)

        output_Q_values_with_action_mask = layers.Multiply(name='output_Q_values_with_action_mask')(
            [output_Q_values, action_mask])

        model = Model(inputs=[state_input, action_mask], outputs=[output_Q_values_with_action_mask])

        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.learning_rate))

        return model

    def start(self):

        for episode in range(self.config.NUMBER_OF_TRAINING_EPISODES):
            self.frames_memory.clear()

            self.env.reset()

            current_image = self.env.render(mode='rgb_array')

            current_frame = self.process_image(current_image)  # the frame is an greyscale image of the current position
            current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])

            current_state = np.repeat(current_frame, self.config.FRAMES_IN_STATE, axis=0)

            self.frames_memory.extend(current_state)

            self.do_learn(current_state, episode)

        # when all the trainign is finished and all loss values are gathered write them in the file
        with open('./loss_values_during_training.csv', mode='a+', newline='') as numerical_data:
            numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
            numerical_data_writer.writerow(self.training_model_history.history['loss'])

    def do_learn(self, current_state, episode):

        reward_sum = 0

        for time_step in range(self.config.TIME_STEPS_IN_EPISODE):

            # calculate a new action only when all frames from a state had been changed
            if time_step % self.config.FRAMES_IN_STATE == 0:
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
            if len(self.replay_buffer) == self.config.MINIMUM_SAMPLES_TO_START_TRAINING:
                self.training = True

            if self.training:
                self.train_training_network()

            reward_sum += reward
            current_state = new_state

            if done:
                break

        if time_step >= self.config.TIME_STEPS_IN_EPISODE - 1:
            print("Failed to finish task in episode {} with reward {} and epsilon {}".format(episode, reward_sum,
                                                                                             self.epsilon))
        else:
            print("Success in epsoide {}, used {} time steps!".format(episode, time_step))
            self.train_network.save(
                cts.Constants.PATH_TO_MODELS_SUCCESSFUL + 'successful_while_training_episode_{}_timesteps_{}_rewards_{}.h5'.format(
                    episode, time_step, reward_sum))

        if self.training:

            # synchronize model_network and target_network
            if (episode % self.config.UPDATE_TRAINING_NET_THRESHOLD) == 0:
                self.synch_networks()

            # save weights for tracking progress
            if (episode % self.config.SAVE_MODEL_THRESHOLD) == 0:
                print('Data saved at episode:', episode)
                self.train_network.save(
                    cts.Constants.PATH_TO_MODELS_TRACKING_PROGRESS_TRESHOLD_SAVES + 'DQN_CNN_model_episode_{}.h5'.format(
                        episode, episode))
            with open('./rewards_in_episodes.csv', mode='a+', newline='') as numerical_data:
                numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                numerical_data_writer.writerow([episode, reward_sum, 1 if done else 0])

    def get_best_action(self, state):
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
            action = np.argmax(self.train_network.predict([state, np.ones((1, self.config.NUMBER_OF_ACTIONS))]))

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


        # the values of the next state the agent arrives used to update it's Q_value for the (current_state, action) from the sample
        next_state_Q_values = self.target_network.predict(
            [new_states, np.repeat(np.ones((1, self.config.NUMBER_OF_ACTIONS)), self.config.BATCH_SIZE, axis=0)])

        # create the targets for the case when the final state is not terminal
        updated_Q_values = rewards + self.gamma * next_state_Q_values.max(axis=1)

        # if the final state is terminal than the pre-terminal state(current state) has Q = reward only
        updated_Q_values[dones] = rewards[dones]

        encoded_actions = to_categorical(actions, num_classes=self.config.NUMBER_OF_ACTIONS)

        updated_Q_values = encoded_actions * updated_Q_values[:, None]
        updated_Q_values = np.asarray(updated_Q_values, dtype=float)

        self.train_network.fit([current_states, encoded_actions], updated_Q_values, epochs=1, verbose=0, callbacks=[self.training_model_history])

        # with open('./loss_values_during_training.csv', mode='a+', newline='') as numerical_data:
        #     numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
        #                                        quoting=csv.QUOTE_MINIMAL)
        #     numerical_data_writer.writerow([self.training_model_history.history['loss'][-1]])

    def get_samples_batch(self):

        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return

        samples = random.sample(self.replay_buffer, self.config.BATCH_SIZE)

        return samples


    def synch_networks(self):
        self.target_network.set_weights(self.train_network.get_weights())

    def process_image(self, image):
        # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.resize(image, (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH))

    def normalize_images(self, image):
        return np.float32(np.true_divide(image, 255))
