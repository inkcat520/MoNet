import os
from utils.data_utils import *


class h36m:
    exp = "h36m"
    subjects = [1, 5, 6, 7, 8, 9, 11]
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]

    train_subject = [1, 6, 7, 8, 9, 11]
    eval_subject = [5]

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def define_acts(self, action):
        """
        Define the list of actions we are using.
        Args
          action: String with the passed action. Could be "all"
        Returns
          actions: List of strings of actions
        Raises
          ValueError if the action is not included in H3.6M
        """
        if action in self.actions:
            return [action]

        if action == "all":
            return self.actions

        if action == "all_srnn":
            return ["walking", "eating", "smoking", "discussion"]

        raise ValueError("Unrecognized action: {}".format(action))

    def load_data(self, subjects, actions):
        dataset = {}
        actions = self.define_acts(actions)
        for subject in subjects:
            for action in actions:
                for subact in [1, 2]:  # subactions
                    print("Reading subject {0}, action {1}, subaction {2}".format(subject, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.data_dir, subject, action, subact)
                    action_sequence = read_series(filename)  # N*99
                    even_list = range(0, action_sequence.shape[0], 2)
                    dataset[(subject, action, subact, 'even')] = action_sequence[even_list]
                    even_list = range(1, action_sequence.shape[0], 2)
                    dataset[(subject, action, subact, 'odd')] = action_sequence[even_list]

        return dataset

    @staticmethod
    def get_data(data, sub, act, subact):
        dataset = {(sub, act, subact): data[(sub, act, subact, 'even')]}

        return dataset

    @staticmethod
    def get_xyz(data):
        data_xyz = {}
        for key, value in data.items():
            the_sequence = expmap2xyz_torch(value).reshape(len(value), -1)
            data_xyz[key] = the_sequence.cpu().numpy()
        return data_xyz

    @staticmethod
    def get_tri_xyz(data):
        data_tri_xyz = {}
        for key, value in data.items():
            the_sequence = tri_xyz_torch(value).reshape(len(value), -1)
            data_tri_xyz[key] = the_sequence.cpu().numpy()
        return data_tri_xyz

    @staticmethod
    def get_wel(data):
        data_trans = {}
        for key, value in data.items():
            the_sequence = wel(value).reshape(len(value), -1)
            data_trans[key] = the_sequence
        return data_trans

    @staticmethod
    def get_angle(data):
        data_trans = {}
        for key, value in data.items():
            the_sequence = expmap2angle_torch(value).reshape(len(value), -1)
            data_trans[key] = the_sequence.cpu().numpy()
        return data_trans

