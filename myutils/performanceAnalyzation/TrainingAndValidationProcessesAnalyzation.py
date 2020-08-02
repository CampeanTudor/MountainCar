import numpy as np
import csv

import matplotlib.pyplot as plt

from collections import deque
import gym
from keras import models

from myutils.ModelTrainingAndValidationParameters import ModelTrainingAndValidationParameters
from myutils.trainingClasses.MountainCarConvolutionalTraining import MountainCarConvolutionalTraining as conv_trainor, \
    MountainCarConvolutionalTraining
import numpy as np
import myutils.constants.Constants as cts
from myutils.gym_custom.gym_custom import MountainCarEnvWrapper
import csv

import os


class TrainingAndValidationProcessesAnalyzation:

    def __init__(self):

        self.config = ModelTrainingAndValidationParameters()

    def generate_plot_of_reward_accumulation_during_training(self):

        a = []

        with open('./rewards_in_episodes.csv', mode='r', newline='') as numerical_data:
            reader = csv.reader(numerical_data, delimiter=',')
            for row in reader:
                a.extend([np.asarray([float(row[0]), float(row[1]), float(row[2])])]),

        a = np.asarray(a)
        episodes = a[:, 0]
        rewards = a[:, 1]
        episodes_won = a[:, 2]

        number_of_points_on_x_axis = 21
        training_episodes = self.config.NUMBER_OF_TRAINING_EPISODES


        plt.figure(figsize=(17, 4))
        plt.plot(episodes, rewards, label="rewards/episode")
        plt.plot(episodes, episodes_won, label="episode won")
        plt.xlabel('episodes')
        plt.ylabel('total reward/episode')
        plt.legend(loc="upper left", bbox_to_anchor=(0, 1.18))
        plt.xticks(np.linspace(0, training_episodes, number_of_points_on_x_axis))
        plt.savefig("training_process_plot.png".format(training_episodes))


    def generate_plot_of_loss_evolution_during_training(self):

        loss_values = []


        with open('./loss_values_during_training.csv', mode='r', newline='') as numerical_data:
            reader = csv.reader(numerical_data, delimiter=',')
            for row in reader:
                loss_values.extend([np.asarray(float(row[0]))]),

        loss_values = np.asarray(loss_values)

        points_on_x_axis = 31

        plt.figure(figsize=(40, 4))
        plt.plot(loss_values, label="training net loss value/time step in every episode")

        plt.xlabel('episodes')
        plt.ylabel('loss value for each time step')
        plt.legend(loc="upper left", bbox_to_anchor=(0, 1.18))

        plt.margins(x=0.005)
        plt.ylim(0, 0.00025)

        x = np.linspace(0, len(loss_values), points_on_x_axis)
        real_episodes_x_ticks = np.linspace(0, self.config.NUMBER_OF_TRAINING_EPISODES, points_on_x_axis)
        plt.xticks(x, real_episodes_x_ticks)
        plt.savefig("loss_function_plot_in_training_process.png")

    def generate_data_for_validation_plot(self):

        conv_trainor = MountainCarConvolutionalTraining()

        env = self.config.ENVIRONMENT

        models_from_training = os.listdir(cts.Constants.PATH_TO_MODELS_TRACKING_PROGRESS_TRESHOLD_SAVES)

        number_of_models_saved = len(models_from_training)

        for current_model_tested in range(number_of_models_saved):

            model = models.load_model(
                cts.Constants.PATH_TO_MODELS_TRACKING_PROGRESS_TRESHOLD_SAVES + models_from_training[
                    current_model_tested])


            frames_memory = deque(maxlen=self.config.FRAMES_IN_STATE)

            for ep in range(self.config.NUMBER_OF_EVALUATION_EPISODES_PER_MODEL):

                env.reset()

                reward_sum = 0

                current_image = env.render(mode='rgb_array')

                # the frame is an greyscale image of the current position
                current_frame = conv_trainor.process_image(current_image)
                current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])

                current_state = np.repeat(current_frame, self.config.FRAMES_IN_STATE, axis=0)

                frames_memory.extend(current_state)

                for t in range(self.config.TIME_STEPS_IN_EPISODE):

                    if (t % self.config.FRAMES_SKIP) == 0:
                        current_state = current_state.reshape(1, current_state.shape[0], current_state.shape[1], current_state.shape[2])
                        best_action = np.argmax(model.predict([current_state, np.ones((1, self.config.NUMBER_OF_ACTIONS))]))

                    new_state_numerical, reward, done, _ = env.step_with_custom_reward(best_action)

                    new_image = env.render(mode='rgb_array')
                    next_frame = conv_trainor.process_image(new_image)
                    next_frame = next_frame.reshape(next_frame.shape[0], next_frame.shape[1])

                    # current_state is a FIFO buffer so just by appending the size  of current_state is constant
                    frames_memory.append(next_frame)

                    new_state = np.asarray(frames_memory)

                    # make the training possible only when the minimum experience was gathered

                    reward_sum += reward
                    current_state = new_state

                    if done:
                        break


                with open('./test_during_training_analysis.csv', mode='a+', newline='') as numerical_data:
                    numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"',
                                                       quoting=csv.QUOTE_MINIMAL)
                    numerical_data_writer.writerow([ep, reward_sum, 1 if done else 0])


    def generate_plot_of_reward_accumulation_during_validation(self):

        a = []
        number_of_validation_eps_for_model = self.config.NUMBER_OF_EVALUATION_EPISODES_PER_MODEL
        #number_of_models = int(self.config.NUMBER_OF_TRAINING_EPISODES/self.config.SAVE_MODEL_THRESHOLD) - 1
        number_of_models = 9 #for this case only

        with open('./test_during_training_analysis.csv', mode='r', newline='') as numerical_data:
            reader = csv.reader(numerical_data, delimiter=',')
            for row in reader:
                a.extend([np.asarray([float(row[0]), float(row[1]), float(row[2])])]),

        a = np.asarray(a)
        episodes = a[:, 0]

        # because for each new model the episodes start from 0, in order to include all validation data for all the models in 1 graph,
        # I changed the validation episodes for each new model to start instead of  0 with 0+current_model*100(if the data is generated for 100 eps/model)
        for i in range(number_of_models):
            a[i * number_of_validation_eps_for_model:(i + 1) * number_of_validation_eps_for_model, 0] = \
                a[i * number_of_validation_eps_for_model:(i + 1) * number_of_validation_eps_for_model, 0] + np.ones(
                    number_of_validation_eps_for_model) * i * number_of_validation_eps_for_model

        rewards = a[:, 1]
        episodes_won = a[:, 2]


        plt.figure(figsize=(17, 4))
        plt.plot(episodes, rewards, label="rewards/episode")
        plt.plot(episodes, episodes_won, label="episode won")
        plt.xlabel('episodes')
        plt.ylabel('total reward/episode')
        plt.legend(loc="upper left", bbox_to_anchor=(0, 1.18))

        # used xticks to include all the validation data -> 900 eps in this case, and to index where the data for the new model starts -> 10 points for 9 intervals
        plt.xticks(np.linspace(0, 900, 10))
        plt.savefig("validation_process_plot.png")
        print("Done")



    def validation_process(self):

        env = self.config.ENVIRONMENT

        episodes = 20

        # load the network
        model = models.load_model(
            cts.Constants.PATH_TO_MODELS_SUCCESSFUL + 'successful_while_training_episode_1062_timesteps_108_rewards_9.372071213400712.h5')



        frames_memory = deque(maxlen=self.config.FRAMES_IN_STATE)

        for i in range(episodes):

            env.reset()

            reward_sum = 0

            current_image = env.render(mode='rgb_array')
            current_frame = conv_trainor.process_image(conv_trainor,
                                                       current_image)  # the frame is an greyscale image of the current position
            current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])
            current_state = np.repeat(current_frame, self.config.FRAMES_IN_STATE, axis=0)
            frames_memory.extend(current_state)

            for t in range(self.config.TIME_STEPS_IN_EPISODE):

                if (t % self.config.FRAMES_SKIP) == 0:
                    current_state = current_state.reshape(1, current_state.shape[0], current_state.shape[1],
                                                          current_state.shape[2])
                    best_action = np.argmax(model.predict([current_state, np.ones((1, 3))]))

                new_state_numerical, reward, done, _ = env.step(best_action)

                new_image = env.render(mode='rgb_array')
                next_frame = conv_trainor.process_image(conv_trainor, new_image)
                next_frame = next_frame.reshape(next_frame.shape[0], next_frame.shape[1])

                # current_state is a FIFO buffer so just by appending the size  of current_state is constant
                frames_memory.append(next_frame)

                new_state = np.asarray(frames_memory)

                # make the training possible only when the minimum experience was gathered

                reward_sum += reward
                current_state = new_state

                if done:
                    break

            if t >= self.config.TIME_STEPS_IN_EPISODE - 1:
                print("Failed to finish task in episode {} with reward {} ".format(i, reward_sum))
            else:
                print("Success in epsoide {}, used {} time steps and obtain reward_sum : {}!".format(i, t, reward_sum))
