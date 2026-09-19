import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from cs231n.image_utils import image_from_url


class ImageFromUrlTest(unittest.TestCase):
    def test_local_file_url_can_be_read_and_temporary_file_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            expected = np.zeros((2, 3, 3), dtype=np.uint8)
            imageio.imwrite(source, expected)

            actual = image_from_url(source.as_uri())

            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
