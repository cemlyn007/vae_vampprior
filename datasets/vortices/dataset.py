import os
import pickle

import pyvista as pv
import torch
from torch.utils.data.dataset import Dataset


class VoxelDataset(Dataset):

    def __init__(self, load_path, data_only=False):
        super(VoxelDataset, self).__init__()
        self.load_path = load_path
        self.filepaths = self.get_filepaths()
        self.data_only = data_only

    def get_filepaths(self):
        if os.path.exists(self.load_path):
            return tuple([os.path.join(self.load_path, fn) for fn in os.listdir(self.load_path)])
        else:
            raise OSError("Load path doesn't exist")

    def load(self, filepath: str):
        # Load the specifc filepaths here
        mesh = pv.read(filepath)
        image = torch.tensor(mesh["vtkValidPointMask"].reshape(mesh.dimensions, order='F')).unsqueeze(0).float()
        vortid = torch.tensor(int(filepath.split("_")[-1].split(".")[0])).int()
        return image, vortid

    def _save_data_with_pickle(self, filepath, data):
        """
        Pickle dump the data into a file given the filepath
        :param filepath: filepath where the data will be dumped (endswith ".pickle")
        :param data: data to be pickled
        :return: None
        """
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_sample_from_pickle(self, filepath):
        """
        Loads the (graph, target) given the filepath
        :param filepath: location to load data from
        :return: contents, usually of the form (graph, target)
        """
        with open(filepath, "rb") as f:
            contents = pickle.load(f)
        return contents

    @staticmethod
    def check_tensor(tensor):
        """
        Check if the tensor contains any nan or inf values. If so it will raise an error
        :param tensor: tensor to be checked
        :return:
        """
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise ZeroDivisionError(f"Normalising went wrong, contains nans or infs")

    def __getitem__(self, item):
        if self.data_only:
            return self.load(self.filepaths[item])[0]
        else:
            return self.load(self.filepaths[item])

    def __len__(self):
        return len(self.filepaths)


if __name__ == "__main__":

    # normalised in the sense that the voxelised meshes have been centered and scaled to fit within a unit cube
    load_path = os.path.join(os.getcwd(), "data", "voxelised", "normalised")
    # directory to store voxelised data that has had each pixel normalised
    # normalised_dir = os.path.join(os.getcwd(), "data", "voxelised", "pixel_normalised")

    dataset = VoxelDataset(load_path=load_path)

    for img in dataset:
        print(img)
