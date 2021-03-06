import numpy as np
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch
from torchvision.datasets.folder import default_loader
import os


class SiameseDataset_original(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, base_dataset):
        super(SiameseDataset_original, self).__init__()
        self.dataset = base_dataset
        self.transform = self.dataset.transform

        self.labels = np.array(self.dataset.imgs)[:, 1]
        self.data = np.array(self.dataset.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.data[siamese_index]

        img1 = default_loader(img1)
        img2 = default_loader(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)

class SiameseDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, root, transform):
        super(SiameseDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.data[siamese_index]

        img1 = default_loader(img1)
        img2 = default_loader(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.imgs)

# class SggDataset_4images(Dataset):
class SggDataset_original(Dataset):
    """
    Train: For each sample creates randomly 4 images
    Test: Creates fixed pairs for testing
    """

    def __init__(self, base_dataset):
        super(SggDataset_original, self).__init__()
        self.dataset = base_dataset
        self.transform = self.dataset.transform

        self.train_labels = np.array(self.dataset.imgs)[:, 1].astype(int)
        self.train_data = np.array(self.dataset.imgs)[:, 0]
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        img_num = 4
        label = self.train_labels[index].item()
        img, label = self.__getimgs_bylabel__(label, img_num)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def __getimgs_bylabel__(self, label, img_num):
        if len(self.label_to_indices[label]) >= img_num:
            index = np.random.choice(self.label_to_indices[label], size=img_num, replace=False)
        else:
            index1 = np.random.choice(self.label_to_indices[label], size=len(self.label_to_indices[label]), replace=False)
            index2 = np.random.choice(self.label_to_indices[label], size=img_num - len(self.label_to_indices[label]),
                                      replace=True)
            index = np.concatenate((index1, index2))
        for i in range(img_num):
            img_temp = (self.train_data[index[i]])
            label_temp = (self.train_labels[index[i]])
            if type(label_temp) not in (tuple, list):
                label_temp = (label_temp,)
            label_temp = torch.LongTensor(label_temp)
            img_temp = default_loader(img_temp)
            if self.transform is not None:
                img_temp = self.transform(img_temp)
                img_temp = img_temp.unsqueeze(0)
            if i == 0:
                img = img_temp
                label = label_temp
            else:
                img = torch.cat((img, img_temp), 0)
                label = torch.cat((label, label_temp), 0)

        return img, label


class SggDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly 4 images
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform):
        super(SggDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1].astype(int)
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        cams = []
        for s in self.imgs:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img_num = 4
        label = self.labels[index].item()
        img, label = self.__getimgs_bylabel__(label, img_num)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __getimgs_bylabel__(self, label, img_num):
        if len(self.label_to_indices[label]) >= img_num:
            index = np.random.choice(self.label_to_indices[label], size=img_num, replace=False)
        else:
            index1 = np.random.choice(self.label_to_indices[label], size=len(self.label_to_indices[label]), replace=False)
            index2 = np.random.choice(self.label_to_indices[label], size=img_num - len(self.label_to_indices[label]),
                                      replace=True)
            index = np.concatenate((index1, index2))
        for i in range(img_num):
            img_temp = (self.data[index[i]])
            label_temp = (self.labels[index[i]])
            if type(label_temp) not in (tuple, list):
                label_temp = (label_temp,)
            label_temp = torch.LongTensor(label_temp)
            img_temp = default_loader(img_temp)
            if self.transform is not None:
                img_temp = self.transform(img_temp)
                img_temp = img_temp.unsqueeze(0)
            if i == 0:
                img = img_temp
                label = label_temp
            else:
                img = torch.cat((img, img_temp), 0)
                label = torch.cat((label, label_temp), 0)

        return img, label


class SggDataset_48_4images(Dataset):
    """
    Train: For each sample creates randomly 48*4 images
    Test: Creates fixed pairs for testing
    """

    def __init__(self, base_dataset):
        super(SggDataset_48_4images, self).__init__()
        self.dataset = base_dataset
        self.transform = self.dataset.transform

        self.labels = np.array(self.dataset.imgs)[:, 1].astype(int)
        self.data = np.array(self.dataset.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}


    def __getitem__(self, index):
        id_num = 2
        img_num = 3
        label1 = self.labels[index]
        label2 = np.random.choice(list(self.labels_set - set([label1])), id_num - 1, replace=False)
        label = np.concatenate(([label1], label2), 0)
        for i in range(id_num):
            img_temp, label_temp = self.__getimgs_bylabel__(label[i], img_num)
            if i == 0:
                img = img_temp.unsqueeze(0)
                label_result = label_temp.unsqueeze(0)
            else:
                img = torch.cat((img, img_temp.unsqueeze(0)), 0)
                label_result = torch.cat((label_result, label_temp.unsqueeze(0)), 0)

        return img, label_result

    def __len__(self):
        return len(self.dataset)

    def __getimgs_bylabel__(self, label, img_num):
        if len(self.label_to_indices[label]) >= img_num:
            index = np.random.choice(self.label_to_indices[label], size=img_num, replace=False)
        else:
            index1 = np.random.choice(self.label_to_indices[label], size=len(self.label_to_indices[label]), replace=False)
            index2 = np.random.choice(self.label_to_indices[label], size=img_num - len(self.label_to_indices[label]),
                                      replace=True)
            index = np.concatenate((index1, index2))
        for i in range(img_num):
            img_temp = (self.data[index[i]])
            label_temp = (self.labels[index[i]])
            if type(label_temp) not in (tuple, list):
                label_temp = (label_temp,)
            label_temp = torch.LongTensor(label_temp)
            img_temp = default_loader(img_temp)
            if self.transform is not None:
                img_temp = self.transform(img_temp)
                img_temp = img_temp.unsqueeze(0)
            if i == 0:
                img = img_temp
                label = label_temp
            else:
                img = torch.cat((img, img_temp), 0)
                label = torch.cat((label, label_temp), 0)

        return img, label



class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
