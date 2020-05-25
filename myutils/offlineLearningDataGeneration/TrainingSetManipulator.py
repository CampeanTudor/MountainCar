import numpy as np
import cv2
import csv
import myutils.constants.Constants as cts


class TrainingSetManipulator:


    def save_state_to_file(self, sample):
        current_state = sample[0]
        best_action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        done = sample[4]

        f = open('put_file_path', 'a+')
        f.write("State: \n")
        np.savetxt(f, current_state[0], header='Current_state[{}]'.format(0), delimiter=',')
        np.savetxt(f, np.array([best_action, reward, done]), header='Best action, Reward, Done', delimiter=',')
        np.savetxt(f, next_state[0], header='Next_state[{}]'.format(0), delimiter=',')
        f.write("\n------------\n")

    def save_data_jpg_and_vcs(self, current_state, next_state, numerical_array, sample_number):
        current_state_1, current_state_2 = np.split(current_state, 2, axis=0)
        next_state_1, next_state_2 = np.split(next_state, 2, axis=0)

        cv2.imwrite(cts.Constants.PATH_TO_TRAINING_CURRENT_STATE_SAMPLE_TEMPLATE.format(0, sample_number), current_state_1.reshape(100, 150))
        cv2.imwrite(cts.Constants.PATH_TO_TRAINING_CURRENT_STATE_SAMPLE_TEMPLATE.format(1, sample_number), current_state_2.reshape(100, 150))

        cv2.imwrite(cts.Constants.PATH_TO_TRAINING_NEXT_STATE_SAMPLE_TEMPLATE.format(0, sample_number), next_state_1.reshape(100, 150))
        cv2.imwrite(cts.Constants.PATH_TO_TRAINING_NEXT_STATE_SAMPLE_TEMPLATE.format(1, sample_number), next_state_2.reshape(100, 150))

        with open(cts.Constants.PATH_TO_TRAINING_NEXT_STATE_SAMPLE_TEMPLATE, mode='a+', newline='') as numerical_data:
            numerical_data_writer = csv.writer(numerical_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            numerical_data_writer.writerow(numerical_array)
