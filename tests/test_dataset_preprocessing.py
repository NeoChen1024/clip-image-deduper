import os
import unittest

from clip_training.dataset_preprocessing import output_relative_image_path


class DatasetPreprocessingTests(unittest.TestCase):
    def test_output_relative_image_path_preserves_directory_structure(self):
        nested = output_relative_image_path(os.path.join("a", "b.jpg"))
        flat = output_relative_image_path("a_b.jpg")
        self.assertNotEqual(nested, flat)
        self.assertEqual(nested, os.path.join("images", "a", "b.jpg"))
