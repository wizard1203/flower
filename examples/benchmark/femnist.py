from __future__ import print_function

import csv
import os
import os.path
import warnings

from PIL import Image

from torchvision import transforms

# from .divide_data import DataPartitioner, select_dataset

from divide_data import DataPartitioner, select_dataset

class FEMNIST():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False):

        self.data_file = dataset  # 'train', 'test', 'validation'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)

        # load data and targets
        self.data, self.targets = self.load_file(self.path)
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}

        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.root, imgName))

        # avoid channel error
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    datas.append(row[1])
                    labels.append(int(row[-1]))
                line_count += 1

        return datas, labels

    def load_file(self, path):

        # load meta file to get labels
        datas, labels = self.load_meta_data(os.path.join(
            self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))

        return datas, labels


def get_femnist(path_to_data="./data"):

    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = FEMNIST(
        path_to_data, dataset='train', transform=train_transform)
    test_dataset = FEMNIST(
        path_to_data, dataset='test', transform=test_transform)
    return train_dataset, test_dataset





def init_femnist(args):

    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = FEMNIST(
        args.data_dir, dataset='train', transform=train_transform)
    test_dataset = FEMNIST(
        args.data_dir, dataset='test', transform=test_transform)

    training_sets = DataPartitioner(
        data=train_dataset, args=args, numOfClass=args.num_class)
    training_sets.partition_data_helper(
        num_clients=args.num_participants, data_map_file=args.data_map_file)

    testing_sets = DataPartitioner(
        data=test_dataset, args=args, numOfClass=args.num_class, isTest=True)
    testing_sets.partition_data_helper(num_clients=1)

    return testing_sets, testing_sets














