import os
import torch
import logging
import torchvision
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# dataset wrapper module
class CINIC10(torchvision.datasets.ImageFolder):
    base_folder = 'cinic-10-batches-py'
    zip_md5 = '6ee4d0c996905fe93221de577967a372'
    splits = ('train', 'val', 'test')
    filename = 'CINIC-10.tar.gz'
    url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'

    def __init__(self, root, split='train', download=True,  **kwargs):
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_transforms = [
            transforms.ToTensor(),
            normalize,
        ]
        self.data_root = os.path.expanduser(root)
        self.split = torchvision.datasets.utils.verify_str_arg(split, 'split', self.splits)
        if download:
            self.download()
        if not self._check_exists():
            err = 'Dataset not found or corrupted. You can use download=True to download it'
            logger.exception(err)
            raise RuntimeError(err)

        super().__init__(root=self.split_folder, transform=train_transforms, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def download(self):
        if self._check_exists():
            return
        torchvision.datasets.utils.download_and_extract_archive(
            self.url, self.dataset_folder, filename=self.filename,
            remove_finished=True, md5=self.zip_md5
        )

    def __repr__(self):
        rep_str = {'train': 'CLIENT', 'test': 'SERVER'}
        return f'[CINIC10] {rep_str[self.split]}'


# helper method to fetch CINIC-10 dataset
def fetch_cinic10( root):
    logger.info('[LOAD] [CINIC10] Fetching dataset!')

    # default arguments
    raw_train = CINIC10(root, split='train')


    raw_test = CINIC10(root, split='test')

    logger.info('[LOAD] [CINIC10] ...fetched dataset!')


    return raw_train, raw_test


def get_cinic10(path):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    transform = [
        transforms.ToTensor(),
        normalize,
    ]


    trainset = torchvision.datasets.ImageFolder(root=f'{path}/train', transform=transforms.Compose(transform))
    testset = torchvision.datasets.ImageFolder(root=f'{path}/test', transform=transforms.Compose(transform))


    return trainset,testset


