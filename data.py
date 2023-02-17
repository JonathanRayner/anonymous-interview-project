#!/usr/bin/env python3
from pathlib import Path
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


def load_image(path_image: str) -> Image.Image:
    """Load image from harddrive and return 3-channel PIL image.
    Args:
        path_image (str): image path
    Returns:
        Image.Image: loaded image
    """
    return Image.open(path_image).convert('RGB')


def get_person_image_paths(path_person_set: str) -> dict:
    """Creates mapping from person name to list of images.
    Args:
        path_person_set (str): Path to dataset that contains folder of images.
    Returns:
        Dict[str, List]: Mapping from person name to image paths,
                         For instance {'name': ['/path/image1.jpg', '/path/image2.jpg']}
    """
    path_person_set = Path(path_person_set)
    person_paths = filter(Path.is_dir, path_person_set.glob('*'))
    return {
        path.name: list(path.glob('*.jpg')) for path in person_paths
    }


def get_persons_with_at_least_k_images(person_paths: dict, k: int) -> list:
    """Filter persons and return names of those having at least k images
    Args:
        person_paths (dict): dict of persons, as returned by `get_person_image_paths`
        k (int): number of images to filter for

    Returns:
        list: list of filtered person names
    """
    return [name for name, paths in person_paths.items() if len(paths) >= k]


class TripletFaceDataset(Dataset):

    def __init__(self, path) -> None:
        super().__init__()

        self.person_paths = get_person_image_paths(path)
        self.persons = list(self.person_paths.keys())  # changed, so we can sample from this
        self.persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """
        # sample 2 positive paths without replacement
        person = self.persons_positive[index]
        anchor, positive = random.sample(self.person_paths[person], k=2)

        # sample a negative person
        negative_person = random.choice(self.persons)

        # keep sampling as needed until we get a negative person
        while negative_person == person:
            negative_person = random.choice(self.persons)
        negative = random.choice(self.person_paths[negative_person])

        return anchor, positive, negative

    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n = self.get_anchor_positive_negative_paths(index)
        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n))
        )

    def __len__(self):
        return len(self.persons_positive)


if __name__ == "__main__":
    # This file is supposed to be imported, but you can run it do perform some unittests
    # or other investigations on the dataloading.
    import argparse
    import unittest
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', type=Path)
    args = parser.parse_args()

    class DatasetTests(unittest.TestCase):
        def setUp(self):
            self.dataset = TripletFaceDataset(args.path_data)

        def test_same_shapes(self):
            index = random.randint(0, len(self.dataset) - 1)
            a, p, n = self.dataset[index]
            self.assertEqual(a.shape, p.shape, 'inconsistent image sizes')
            self.assertEqual(a.shape, n.shape, 'inconsistent image sizes')

        def test_triplet_paths(self):
            index = random.randint(0, len(self.dataset) - 1)
            a, p, n = self.dataset.get_anchor_positive_negative_paths(index)
            self.assertEqual(a.parent.name, p.parent.name)
            self.assertNotEqual(a.parent.name, n.parent.name)

    unittest.main(argv=[''], exit=False)
