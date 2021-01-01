import os
import json
import random

from collections import namedtuple

ListData = namedtuple('ListData', ['id', 'label', 'path'])


class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """

    def __init__(self, dataset_name, json_path_input, json_path_labels, data_root,
                 extension, is_test=False, subset_rate=1):
        self.dataset_name = dataset_name
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels

        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)

        if self.dataset_name == 'charades':
            assert subset_rate == 1, 'charades can not be sampled into mini_dataset, thus subset_rate must be 1.0'
        # prepare a subset of json annos for the method, subset_rate added by Mr. Yan
        self.json_data = self.read_json_input(subset_rate)

    ############## added by Mr. Yan for subset the original dataset by class ###############

    def get_class_dict(self, data):
        class_dict = {}
        for idx, elem in enumerate(data):
            label = self.clean_template(elem['template'])
            if label in class_dict.keys():
                class_dict[label].append(idx)
            else:
                class_dict[label] = [idx]
        return class_dict

    def sample_data_by_class(self, data, rate=0.1):
        subset_data = []
        class_dict = self.get_class_dict(data)
        for k in class_dict.keys():
            sampling_num = int(len(class_dict[k])*rate)+1
            sampled_ids = class_dict[k][:sampling_num]
            subset_data.extend([data[i] for i in sampled_ids])
        return subset_data
    #########################################################################################

    def read_json_input(self, subset_rate=1):  # SUBSET_RATE is added by Mr. Yan
        json_data = []
        if not self.is_test:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                ##################### added by Mr. Yan ###########################
                # to sample the previous {subset_rate}% json data for each class
                if subset_rate != 1.0:
                    json_reader = self.sample_data_by_class(
                        json_reader, subset_rate)
                #################################################################
                for elem in json_reader:
                    if self.dataset_name == 'sth_else':
                        label = self.clean_template(elem['template'])
                        if label not in self.classes:
                            raise ValueError("Label mismatch! Please correct")
                    elif self.dataset_name == 'charades':
                        label = elem['label']
                        for la in label:
                            if la not in self.classes:
                                raise ValueError(
                                    "Label mismatch! Please correct! %s not in class list!" % (la))
                    item = ListData(elem['id'],
                                    label,
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        else:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'r') as jsonfile:
            # print(jsonfile)
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class WebmDataset(DatasetBase):
    def __init__(self, dataset_name, json_path_input, json_path_labels, data_root,
                 is_test=False, subset_rate=1):
        EXTENSION = ".webm"
        super().__init__(dataset_name, json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test, subset_rate)
