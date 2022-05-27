import os
import torch
import requests
import itertools
import torchvision

import numpy as np

class BAS(torch.utils.data.Dataset):
    """ Bars And Stripes (BAS) Synthetic Dataset

    Args:
        rows (int): Number of rows in the BAS image.

        cols (int): Number of columns in the BAS image.

        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version.

        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Example:

        >>> # Using Class method
        >>> import matplotlib.pyplot as plt
        >>> img,target = BAS.generate_bars_and_stripes(4,5)
        >>> fig,axs = plt.subplots(2,len(img)//2)
        >>> for i,ax in enumerate(axs.flat):
        >>>     ms = ax.matshow(img[i],vmin=0, vmax=1)
        >>>     ax.axis('off')
    """

    def __init__(self, rows, cols, embed_label=False,
                 transform=None, target_transform=None):
        self.data, self.targets = self.generate(rows,cols,embed_label)
        self.transform = transform
        self.embed_label = embed_label
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def find(self, item):
        if isinstance(item,torch.Tensor):
            item = item.numpy()

        if isinstance(item,np.ndarray):
            iter = (i for i,d in enumerate(self.data) if np.array_equiv(d,item))
        else:
            raise RuntimeError("Item isn't `torch.Tensor` or `numpy.ndrray`")

        return next(iter,None)

    def __contains__(self, item):
        return bool(self.find(item))

    def score(self, samples):
        """ Given a set of samples, compute the qBAS[1] sampling score:
                qBAS = 2pr/(p+r)
            p: precision or number of correct samples over total samples
            r: recall or number of sampled patterns over total patterns

        Args:
            samples (list or numpy.ndarray): An iterable of numpy.ndarray values
                to be compared to the original data in the dataset.

        Returns:
            precision (float): number of correct samples over total samples

            recall (float): number of sampled patterns over total patterns

            score (float): qBAS score as defined above

        [1] Benedetti, M., et al. A generative modeling approach for
        benchmarking and training shallow quantum circuits. (2019).
        https://doi.org/10.1038/s41534-019-0157-8
        """
        total_samples = len(samples)
        total_patterns = len(self)

        sampled_patterns = [i for i in map(self.find,samples) if i is not None]
        if not sampled_patterns: return 0.0,0.0,0.0

        precision = len(sampled_patterns)/total_samples
        recall = len(set(sampled_patterns))/total_patterns
        score = 2.0*precision*recall/(precision+recall)

        return precision, recall, score

    @classmethod
    def generate(cls, rows, cols, embed_label=False):
        """ Generate the full dataset of rows*cols Bars And Stripes (BAS).
        Args:
            cols (int): number of columns in the generated images

            rows (int): number of rows in the generated images

        Returns:
            data (numpy.ndarray): Array of (rows,cols) images of BAS dataset

            targets (numpy.ndarray): Array of labels for data. Where the empty
                (all zeros), bars (vertical lines), stripes (horizontal lines),
                and full (all ones) images belong to different classes.

        Implementation based on DDQCL project for benchmarking generative models
        with shallow gate-level quantum circuits.

        [1] https://github.com/uchukwu/quantopo
        [2] https://www.nature.com/articles/s41534-019-0157-8
        """
        bars = []
        for h in itertools.product([0., 1.], repeat=cols):
            pattern = np.repeat([h], rows, 0)
            bars.append(pattern)

        stripes = []
        for h in itertools.product([0., 1.], repeat=rows):
            pattern = np.repeat([h], cols, 1)
            stripes.append(pattern.reshape(rows,cols))

        data = np.asarray(np.concatenate((bars[:-1], # ignore all ones
                                         stripes[1:]), # ignore all zeros
                                         axis=0),dtype='float32')

        if embed_label:
            labels  = [(0,0)] # All zeros
            labels += [(0,1)]*(2**cols-2) # Bars
            labels += [(1,0)]*(2**rows-2) # Stripes
            labels += [(1,1)] # All ones
            targets = np.asarray(labels,dtype='float32')
            data[:,-2,-1] = targets[:,0]
            data[:,-1,-1] = targets[:,1]
            return data, targets
        else:
            # Create labels synthetically
            labels  = [0] # All zeros
            labels += [1]*(2**cols-2) # Bars
            labels += [2]*(2**rows-2) # Stripes
            labels += [3] # All ones
            return data, np.asarray(labels,dtype='float32')

class OptDigits(torchvision.datasets.vision.VisionDataset):
    """ Based on the MNIST Dataset implementation, but enough differences to not
        make it a subclass.
    """
    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
    ]

    training_file = ("optdigits.tra", "268ce7771f3f15afbc54402478b1d454")
    test_file = ("optdigits.tes", "a0339c30a8a5312a1b6f9e5c719dcce5")

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train = True,
                 transform = None, target_transform = None,
                 download = True):
        super(OptDigits, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        if download:
            self.download()
        self.data, self.targets = self._load_data()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_data(self):
        filename, _ = self.training_file if self.train else self.test_file
        fpath = os.path.join(self.raw_folder, filename)
        dataset = np.genfromtxt(fpath,delimiter=',',dtype='float32')
        data,targets = np.split(dataset,[64],1)
        return data.reshape((len(data),8,8))/16, targets.flatten()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        filename, md5 = self.training_file if self.train else self.test_file
        fpath = os.path.join(self.raw_folder, filename)

        if torchvision.datasets.utils.check_integrity(fpath,md5):
            print("Using downloaded and verified file " + fpath)
            return

        for mirror in self.mirrors:
            try:
                print('Downloading ' + mirror+filename + ' to ' + fpath)
                with open(fpath, 'wb') as f:
                    response = requests.get(mirror+filename)
                    f.write(response.content)
            except:
                print("Failed download.")
                continue
            if not torchvision.datasets.utils.check_integrity(fpath,md5):
                raise RuntimeError("File not found or corrupted.")
            break
