import os
import torch
import torchvision.transforms as transforms
# from torch.utils.data import Subset
from datasets.dataset import Subset

import torchvision

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == 'LLIE':
        dataset = torchvision.datasets.ImageFolder(
            # os.path.join(args.exp, "datasets", args.path_y),
            args.path_y,
            transform=transforms.Compose([
                transforms.ToTensor()])
        )
        num_items = len(dataset)
        indices = list(range(num_items))
        train_indices, test_indices = (
            indices[: int(num_items * 0.)],
            indices[int(num_items * 0.):],
        )
        test_dataset = Subset(dataset, test_indices)


    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

# import os
# import json
# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# from torchvision import transforms
# from PIL import Image

# def get_dataset_json(args, config):
#     # Define transforms
#     if config.data.random_flip is False:
#         train_transform = test_transform = transforms.Compose(
#             [
#                 transforms.Resize((400,600)),
#                 transforms.ToTensor()
#             ]
#         )
#     else:
#         train_transform = transforms.Compose(
#             [
#                 transforms.Resize(config.data.image_size),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.ToTensor(),
#             ]
#         )
#         test_transform = transforms.Compose(
#             [
#                 transforms.Resize((400,600)),
#                 transforms.ToTensor()
#             ]
#         )

#     # Custom Dataset class for JSON mapping
#     class JSONImageDataset(Dataset):
#         def __init__(self, json_path, transform=None):
#             with open(json_path, 'r') as f:
#                 self.data = json.load(f)
#             self.transform = transform

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             # Get the key-value pair from JSON
#             keys = list(self.data.keys())
#             image_name = keys[idx]
#             target_path = self.data[image_name][0]  # Assuming the first path is the target

#             # Load the images
#             input_image = Image.open(os.path.join('/mnt/KJG/LLIE/Reference/LLIE_un/FourierDiff/exp/datasets/test/low', image_name)).convert('RGB')
#             target_image = Image.open(target_path).convert('RGB')

#             # Apply transforms
#             if self.transform:
#                 input_image = self.transform(input_image)
#                 target_image = self.transform(target_image)

#             return input_image, target_image, image_name

#     if config.data.dataset == 'LLIE':
#         # Create the dataset
#         dataset = JSONImageDataset(
#             json_path='/mnt/KJG/LLIE/RAG_LLIE/dataset/LOLdataset/retrieve_eval_high_LOLv1_1.json',  # Path to the JSON file
#             transform=test_transform
#         )
#         num_items = len(dataset)
#         indices = list(range(num_items))

#         # Split into train and test datasets
#         train_indices, test_indices = (
#             indices[: int(num_items * 0.)],  # 80% for training
#             indices[int(num_items * 0.):],   # 20% for testing
#         )
#         train_dataset = Subset(dataset, train_indices)
#         test_dataset = Subset(dataset, test_indices)

#     else:
#         train_dataset, test_dataset = None, None

#     return train_dataset, test_dataset

