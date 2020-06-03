from collections import deque

import numpy as np
import gym
import cv2
import csv
import random

import myutils.constants.Constants as cts
from myutils.gym_custom.gym_custom import MountainCarEnvWrapper
from myutils.offlineLearningDataGeneration.TrainingSetManipulator import TrainingSetManipulator as TrainingSetManipulator


class OfflineGridTrainingSetGenerator:

    def __init__(self):

        self.file_manipulator = TrainingSetManipulator()
        self.pos_min = -1.2
        self.pos_max = 0.6

        self.vel_min = -0.07
        self.vel_max = 0.07

        self.total_pos_points = 100
        self.total_vel_points = 100

        self.pos_grid = np.linspace(self.pos_min, self.pos_max, self.total_pos_points)
        self.vel_grid = np.linspace(self.vel_min, self.vel_max, self.total_vel_points)
        self.all_action = [0, 1, 2]

        self.samples_deque = deque(maxlen=50) #tune from here the maxlen of the deque

        self.env = MountainCarEnvWrapper(gym.make('MountainCar-v0'))


    def generate_set_of_samples_on_disk(self):


        self.env.reset()

        sample_to_save = 0

        #construct the first state
        for first_state_first_image_position in self.pos_grid:

            # the img differs only for position, is constant for velocity
            first_state_first_image = cv2.imread(cts.Constants.PATH_TO_OFFLINE_LEARNING_STATES_GRID_INDEX_IMAGE1_TEMPLATE.format(self.index_of_value_in_pos_grid(first_state_first_image_position), 0))
            first_state_first_image = self.process_image_before_save(first_state_first_image)

            for first_state_first_image_velocity in self.vel_grid:

                first_state_first_image_numerical = (first_state_first_image_position, first_state_first_image_velocity)

                first_state_second_image_numerical, reward, done, _ = self.env.step_with_hardcoded_values(
                    first_state_first_image_position,
                    first_state_first_image_velocity)

                # approximate the position obtained to the nearest value in the grid
                first_state_second_image_position = self.pos_grid[(np.abs(self.pos_grid - first_state_second_image_numerical[0])).argmin()]

                first_state_second_image_velocity = first_state_second_image_numerical[1] - first_state_first_image_velocity
                first_state_second_image_velocity = self.vel_grid[(np.abs(self.vel_grid - first_state_second_image_velocity).argmin())]

                first_state_second_image = cv2.imread(
                    cts.Constants.PATH_TO_OFFLINE_LEARNING_STATES_GRID_INDEX_IMAGE2_TEMPLATE.format(
                        self.index_of_value_in_pos_grid(first_state_second_image_position),
                        self.index_of_value_in_vel_grid(first_state_second_image_velocity)))

                first_state_second_image = self.process_image_before_save(first_state_second_image)


                first_state_images = np.asarray([first_state_first_image, first_state_second_image])

                #construct the next state
                for state_transition_action in self.all_action:

                    next_state_second_image_numerical, sample_reward, sample_done, _ = self.step(state_transition_action)

                    # approximate the position obtained to the nearest value in the grid
                    second_state_second_image_position = self.pos_grid[(np.abs(self.pos_grid - next_state_second_image_numerical[0])).argmin()]

                    second_state_second_image_velocity = next_state_second_image_numerical[1] - first_state_second_image_velocity
                    second_state_second_image_velocity = self.vel_grid[(np.abs(self.vel_grid - second_state_second_image_velocity)).argmin()]

                    second_state_second_image = cv2.imread(cts.Constants.PATH_TO_OFFLINE_LEARNING_STATES_GRID_INDEX_IMAGE2_TEMPLATE.format(
                        self.index_of_value_in_pos_grid(second_state_second_image_position),
                        self.index_of_value_in_vel_grid(second_state_second_image_velocity)))

                    second_state_second_image = self.process_image_before_save(second_state_second_image)

                    second_state_images = np.asarray([first_state_second_image, second_state_second_image])

                    #save the sample
                    self.save_data_jpg_and_vcs(first_state_images,
                                               second_state_images,
                                               [state_transition_action, sample_reward, sample_done],
                                               sample_to_save)
                    self.env.reset()

                    sample_to_save = sample_to_save + 1

    def generate_batch_of_samples_in_ram(self, size_batch, stack_depth):


        self.env.reset()

        self.samples_deque.clear()

        positions_sampled_from_grid = random.sample(list(self.pos_grid), size_batch)
        velocities_sampled_from_grid = random.sample(list(self.pos_grid), size_batch)

        first_state = deque(maxlen=stack_depth)


        for sample_number in range(size_batch):

            first_state_first_image_position = positions_sampled_from_grid[sample_number]
            first_state_first_image_velocity = velocities_sampled_from_grid[sample_number]

            #set the agent in the position to crate a sample
            self.env.set_state_with_harcoded_values(first_state_first_image_position, first_state_first_image_velocity)

            first_state.clear()


            #obtain the first state
            for frame_number in range(stack_depth):
                current_frame = self.env.render(mode='rgb_array')
                current_frame = self.process_image_before_save(current_frame)
                first_state.append(current_frame)
                self.env.step(random.choice(self.all_action))

            #take action to next state
            action_transition = random.choice(self.all_action)
            second_state_final_frame_numerical, sample_reward, sample_done, _ = self.env.step(action_transition)

            #obtain the last frame for the second state
            second_state_final_frame = self.env.render(mode='rgb_array')
            second_state_final_frame = self.process_image_before_save(second_state_final_frame)

            #create the second state
            second_state = first_state.copy()
            second_state.append(second_state_final_frame)

            first_state_for_sample = np.asarray(first_state)
            second_state_for_sample = np.asarray(second_state)

            #construct the sample
            self.samples_deque.append([first_state_for_sample, action_transition, sample_reward, second_state_for_sample, sample_done])

        return self.samples_deque


    def generarte_grid_of_2_image_states(self):
        env = gym.make('MountainCar-v0')
        env.reset()

        for position in self.pos_grid:
            data_1 = env.step_with_hardcoded_values(position, 0)
            img_1 = env.render(mode='rgb_array')
            for velocity in self.vel_grid:
                data = env.step_with_hardcoded_values(position, velocity)
                img_2 = env.render(mode='rgb_array')
                self.save_state_images_jpg(img_1, img_2, (np.where(self.pos_grid == position)[0])[0], (np.where(self.vel_grid == velocity)[0])[0])



    def save_state_images_jpg(self, image_1, image_2, index_position, index_velocity):

        image_1 = self.process_image_before_save(image_1)
        image_2 = self.process_image_before_save(image_2)

        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_STATES_GRID_INDEX_IMAGE1_TEMPLATE.format(index_position, index_velocity),
                    image_1.reshape(100, 150))
        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_STATES_GRID_INDEX_IMAGE2_TEMPLATE.format(index_position, index_velocity),
                    image_2.reshape(100, 150))


    def save_data_jpg_and_vcs(self, current_state, next_state, numerical_array, sample_number):
        current_state_1, current_state_2 = np.split(current_state, 2, axis=0)
        next_state_1, next_state_2 = np.split(next_state, 2, axis=0)

        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_CURRENT_STATE_TEMPLATE.format(0, sample_number), current_state_1.reshape(100, 150))
        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_CURRENT_STATE_TEMPLATE.format(1, sample_number), current_state_2.reshape(100, 150))

        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_NEXT_STATE_TEMPLATE.format(0, sample_number), next_state_1.reshape(100, 150))
        cv2.imwrite(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLE_NEXT_STATE_TEMPLATE.format(1, sample_number), next_state_2.reshape(100, 150))

        with open(cts.Constants.PATH_TO_OFFLINE_LEARNING_SAMPLES_NUMERICAL_VALUES, mode='a+', newline='') as numerical_data:
            numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            numerical_data_writer.writerow(numerical_array)

    def process_image_before_save(self, image):
        # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (150, 100))

    def index_of_value_in_pos_grid(self, value):
        return np.where(self.pos_grid == value)[0][0]

    def index_of_value_in_vel_grid(self, value):
        return np.where(self.vel_grid == value)[0][0]
