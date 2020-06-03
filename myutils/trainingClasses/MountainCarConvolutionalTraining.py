import csv
from collections import deque
import random
import tensorflow as tf
import cv2
from keras import models
from keras import layers
from keras.optimizers import Adam
from myutils.offlineLearningDataGeneration.TrainingSetManipulator import TrainingSetManipulator
from myutils.offlineLearningDataGeneration.OfflineGridTrainingSetGenerator import OfflineGridTrainingSetGenerator
from myutils.offlineLearningDataGeneration.OfflineGridTrainingSetGenerator import OfflineGridTrainingSetGenerator
import myutils.constants.Constants as cts

import numpy as np
from io import StringIO



class MountainCarConvolutionalTraining:

    def __init__(self, env, training_mode='online'):

        self.env = env

        self.stack_depth, self.image_height, self.image_width = self.get_model_input_shape()
        self.num_actions = env.action_space.n

        self.learning_rate = 0.00025
        self.train_network = self.create_network()
        self.target_network = self.create_network()

        self.epsilon = 1
        self.epsilon_decay = 0.000018
        self.epsilon_min = 0.1

        self.frames_memory = deque(maxlen=self.stack_depth)
        self.replay_buffer = deque(maxlen=200000)
        self.minimum_samples_for_training = 150000
        self.num_pick_from_buffer = 32

        self.time_steps_in_episode = 300  # max is 200

        self.episode_num = 10000

        self.training = False

        self.update_weights_threshold = 35
        self.save_model_threshold = 1000

        self.training_set_manipulator = TrainingSetManipulator()

        self.training_mode = training_mode

        self.offline_trainig_set_generator = OfflineGridTrainingSetGenerator()


    def create_network(self):
        input_shape = (self.stack_depth, self.image_height, self.image_width)

        model = models.Sequential()

        model.add(layers.Conv2D(32, (8, 8), strides=4, padding="same", activation='relu', input_shape=input_shape, name='conv_1'))
        model.add(layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv_2'))
        model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv_3'))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu', name='dense_1'))
        model.add(layers.Dense(self.num_actions, activation='linear', name='output'))

        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.learning_rate))

        return model

    def start(self):

        if self.training_mode == 'online':
            for episode in range(self.episode_num):
                self.frames_memory.clear()

                self.env.reset()

                current_image = self.env.render(mode='rgb_array')
                current_frame = self.process_image(current_image)  # the frame is an greyscale image of the current position
                current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])
                current_state = np.repeat(current_frame, self.stack_depth, axis=0)
                self.frames_memory.extend(current_state)

                self.learn_online_from_episode(current_state, episode)

        elif self.training_mode == 'offline':
            self.learn_offline()

    def learn_online_from_episode(self, current_state, episode):

        reward_sum = 0

        for time_step in range(self.time_steps_in_episode):

            #calculate a new action only when all frames from a state had been changed
            if time_step % self.stack_depth == 0:
                best_action = self.get_best_action(current_state)


            new_state_numerical, reward, done, _ = self.env.step(best_action)
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
                    cts.Constants.PATH_TO_MODELS_TRACKING_PROGRESS_TRESHOLD_SAVES+'DQN_CNN_model_episode_{}.h5'.format(
                        episode, episode))
            with open('./rewards_in_episodes.csv', mode='a+', newline='') as numerical_data:
                numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                numerical_data_writer.writerow([episode, reward_sum])


    def learn_offline(self):
        #echivalentul la 10^3 episoade cu 300 de antrenari/episode
        for i in range(3000000):

            self.train_training_network()

            #decay epsilon
            self.epsilon -= self.epsilon_decay

            #sync networks
            if (i % 300) == 0:
                self.synch_networks()
                print("iteration {} equivalent to {} episodes".format(i, int(i/300)))

            #save model every 5000 iterations
            if (i % 5000) == 0:
                self.train_network.save(cts.Constants.PATH_TO_SAVE_MODEL_OFFLINE_LEARNING_AT_ITERATION_TEMPALTE.format(i))

    def offline_learning_random_sampling(self):

        samples = deque(maxlen=self.num_pick_from_buffer)

        current_state = deque(maxlen=self.stack_depth)
        next_state = deque(maxlen=self.stack_depth)
        for i in range(self.num_pick_from_buffer):
            sample_number = random.randrange(1, 300000)

            current_state_img1 = cv2.imread(
                cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_CURRENT_STATE_TEMPLATE.format(0, sample_number))
            current_state_img2 = cv2.imread(
                cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_CURRENT_STATE_TEMPLATE.format(1, sample_number))
            current_state_img1 = self.process_image(current_state_img1)
            current_state_img2 = self.process_image(current_state_img2)

            current_state.append(current_state_img1)
            current_state.append(current_state_img2)

            next_state_img1 = cv2.imread(
                cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_NEXT_STATE_TEMPLATE.format(0, sample_number))
            next_state_img2 = cv2.imread(
                cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_NEXT_STATE_TEMPLATE.format(1, sample_number))
            next_state_img1 = self.process_image(next_state_img1)
            next_state_img2 = self.process_image(next_state_img2)

            next_state.append(next_state_img1)
            next_state.append(next_state_img2)

            with open(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLES_NUMERICAL_VALUES, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, line in enumerate(reader):
                    if i == sample_number:
                        data = line
                        break


            f = StringIO(data[0])
            reader = csv.reader(f, delimiter=',')
            for element in reader:
                action = int(element[0])
                reward = float(element[1])
                done = element[2] == 'True'
            f.close()

            samples.append([current_state, action, reward, next_state, done])

        return samples



    def get_best_action(self, state):
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
            action = np.argmax(self.train_network.predict(state)[0])

        #update epsilon
        if self.training:
            self.epsilon -= self.epsilon_decay

        return action

    def train_training_network(self):

        samples = self.get_samples_batch()

        if not samples:
            return

        current_state = []
        new_states = []
        rewards = []
        actions = []
        dones = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            current_state.append(state)
            new_states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)

        current_state = np.array(current_state)
        current_state = self.normalize_images(current_state)

        new_states = np.array(new_states)
        new_states = self.normalize_images(new_states)

        current_state_Q_values = self.train_network.predict(current_state)
        next_state_Q_values = self.target_network.predict(new_states)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample

            if done:

                next_state_Q_values[i] = np.zeros(self.num_actions)

            Q_future = max(next_state_Q_values[i])
            (current_state_Q_values[i])[action] = reward + Q_future * 0.99
            i += 1

        self.train_network.fit(current_state, current_state_Q_values, epochs=1, verbose=0)

    def get_samples_batch(self):

        if self.training_mode == 'online':

            if len(self.replay_buffer) < self.num_pick_from_buffer:
                return

            samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)

        elif self.training_mode == 'offline':
            #samples = self.offline_learning_random_sampling()
            samples = self.offline_trainig_set_generator.generate_batch_of_samples_in_ram(self.num_pick_from_buffer,self.stack_depth)

        return samples


    def get_model_input_shape(self):
        self.env.reset()
        initial_image_shape = self.env.render(mode='rgb_array').shape
        image_height = 100 #initial_image_shape[0]
        image_width = 150 #initial_image_shape[1]
        stack_depth = 4

        # dimensions are 2 400 600
        return stack_depth, image_height, image_width

    def synch_networks(self):
        self.target_network.set_weights(self.train_network.get_weights())

    def process_image(self, image):
        # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.resize(image, (150, 100))

    def normalize_images(self, image):
        return np.float32(np.true_divide(image, 255))
