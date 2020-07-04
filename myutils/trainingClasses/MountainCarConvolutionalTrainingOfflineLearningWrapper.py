import csv
import datetime
from io import StringIO
from collections import deque
import random
import numpy as np

import cv2

import myutils.constants.Constants as cts
from myutils.trainingClasses.MountainCarConvolutionalTraining import MountainCarConvolutionalTraining


class MountainCarConvolutionalTrainingOfflineLearningWrapper(MountainCarConvolutionalTraining):

    def __init__(self, env):

        super(MountainCarConvolutionalTrainingOfflineLearningWrapper,self).__init__(env)

        self.pos_min = -1.2
        self.pos_max = 0.6

        self.vel_min = -0.07
        self.vel_max = 0.07

        self.total_pos_points = 100
        self.total_vel_points = 100

        self.pos_grid = np.linspace(self.pos_min, self.pos_max, self.total_pos_points)
        self.vel_grid = np.linspace(self.vel_min, self.vel_max, self.total_vel_points)
        self.all_action = [0, 1, 2]

        self.samples_deque = deque(maxlen=50)  # tune from here the maxlen of the deque

        self.offline_learning_save_threshold = 5000

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
            random_action = random.choice(self.all_action)
            for frame_number in range(stack_depth):
                current_frame = self.env.render(mode='rgb_array')
                current_frame = self.process_image_before_save(current_frame)
                first_state.append(current_frame)
                self.env.step(random_action)

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

    #nu e nevoie nici de current_state nici de episode in logica offline
    def do_learn(self, current_state, episode):
        # echivalentul la 10^3 episoade cu 300 de antrenari/episode
        for i in range(3000000):

            self.train_training_network()

            # sync networks
            if (i % self.time_steps_in_episode) == 0:
                self.synch_networks()
                print("iteration {} equivalent to {} episodes".format(i, int(i / self.time_steps_in_episode)))

            # save model every 5000 iterations
            if (i % self.offline_learning_save_threshold) == 0:
                self.train_network.save(
                    cts.Constants.PATH_TO_SAVE_MODEL_OFFLINE_LEARNING_AT_ITERATION_TEMPALTE.format(int(i)))

    def get_samples_batch(self):

        time_start_reset = datetime.datetime.now()
        self.env.reset()
        time_end_reset = datetime.datetime.now()

        time_start_deque_clear = datetime.datetime.now()
        self.samples_deque.clear()
        time_end_deque_clear = datetime.datetime.now()

        time_start_random_sample = datetime.datetime.now()
        positions_sampled_from_grid = random.sample(list(self.pos_grid), self.num_pick_from_buffer)
        time_end_random_sample = datetime.datetime.now()
        velocities_sampled_from_grid = random.sample(list(self.vel_grid), self.num_pick_from_buffer)

        time_start_deque_init = datetime.datetime.now()
        first_state = deque(maxlen=self.stack_depth)
        time_end_deque_init = datetime.datetime.now()

        time_start_construct_all_samples = datetime.datetime.now()
        for sample_number in range(self.num_pick_from_buffer):

            #make sure that the first state deque is empty before constructing the new sample
            first_state.clear()

            first_state_first_image_position = positions_sampled_from_grid[sample_number]
            first_state_first_image_velocity = velocities_sampled_from_grid[sample_number]

            time_start_set_hardcoded_values = datetime.datetime.now()
            #set the agent in a random init position to crate a sample
            self.env.set_state_with_harcoded_values(first_state_first_image_position, first_state_first_image_velocity)
            time_end_set_hardcoded_values= datetime.datetime.now()


            time_start_render= datetime.datetime.now()
            #obtain the first frame
            time_start_render = datetime.datetime.now()
            current_frame = self.env.render(mode='rgb_array')
            time_end_render = datetime.datetime.now()

            time_start_process_image= datetime.datetime.now()
            current_frame = self.process_image(current_frame)
            first_state.append(current_frame)
            time_end_process_image = datetime.datetime.now()

            #obtain the other stack_depth-1 frames for the first state || one unique action for stack_depth frames
            action = random.choice(self.all_action)
            for frame_number in range(self.stack_depth-1):
                self.env.step(action)
                current_frame = self.env.render(mode='rgb_array')
                current_frame = self.process_image(current_frame)
                first_state.append(current_frame)

            first_state_for_sample = np.asarray(first_state)

            #take action to next state
            action_transition = self.get_best_action(first_state_for_sample)
            second_state_final_frame_numerical, sample_reward, sample_done, _ = self.env.step(action_transition)

            #obtain the last frame for the second state
            second_state_final_frame = self.env.render(mode='rgb_array')
            second_state_final_frame = self.process_image(second_state_final_frame)

            time_start_copy_and_append = datetime.datetime.now()
            #create the second state
            second_state = first_state.copy()
            second_state.append(second_state_final_frame)
            time_end_copy_and_append = datetime.datetime.now()

            second_state_for_sample = np.asarray(second_state)

            #construct the sample
            self.samples_deque.append([first_state_for_sample, action_transition, sample_reward, second_state_for_sample, sample_done])

        time_end_construct_all_samples = datetime.datetime.now()

        print("Time values for get_samples_batch: ")
        print("Environment reset: {}".format(time_end_reset-time_start_reset))
        print("dequeue.clear(): {}".format(time_end_deque_clear- time_start_deque_clear))
        print("random sampling: {}".format(time_end_random_sample- time_start_random_sample))
        print("deque initialize: {}".format(time_end_deque_init- time_start_deque_init))
        print("Step with hardcoded values: {}".format(time_end_set_hardcoded_values- time_start_set_hardcoded_values))
        print("Render time: {}".format(time_end_render- time_start_render))
        print("Process image time: {}".format(time_end_process_image- time_start_process_image))
        print("Copy and append: {}".format(time_end_copy_and_append- time_start_copy_and_append))
        print("Construct all samples: {}".format(time_end_construct_all_samples- time_start_construct_all_samples))

        return self.samples_deque

    def offline_learning_random_sampling_from_files(self):

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

        # update epsilon
        self.epsilon -= self.epsilon_decay

        return action


