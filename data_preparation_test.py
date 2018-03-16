"""
Testing functions of data_preparation module
"""
import unittest
import numpy.testing as nptest
import numpy as np
from data_preparation import load_data
from data_preparation import partition_data
from data_preparation import save_data


class TestDataPreparation(unittest.TestCase):
    """
    Tests the functions in the data_preparation module
    """

    def test_load_data(self):
        """
        Tests the load data funciton. required TESTING_data.csv file
        Reads from test file TESTING_data.csv, and verifies contents
        """
        header_orig_data, data_orig_data, _ = load_data(
            './datasets/TESTING_data.csv')

        header_array = np.array(['Item1', 'Item2', 'Item3', 'Item4'])
        # load_data is transposing this item
        header_array = header_array
        nptest.assert_array_equal(header_orig_data, header_array)

        data_array = np.array([[1, 2, 3, 4, ], [5, 6, 7, 8]])
        # load_data is transposing this item
        data_array = data_array
        nptest.assert_array_equal(data_orig_data, data_array)

    def test_partition_data(self):
        """
        Tests the partition data funciton. required TESTING_partition_data.csv file
        """
        _, data_orig_data, _ = load_data('./datasets/TESTING_partition_data.csv')

        train_set, test_set, dev_set = partition_data(data_orig_data)
        self.assertEqual(train_set.shape[0], 6)
        self.assertEqual(test_set.shape[0], 2)
        self.assertEqual(dev_set.shape[0], 2)

        # Test a different partition configuration
        train_set, test_set, dev_set = partition_data(data_orig_data, 0.8, 0.1)
        self.assertEqual(train_set.shape[0], 8)
        self.assertEqual(test_set.shape[0], 1)
        self.assertEqual(dev_set.shape[0], 1)

        # Test a configuration with no dev
        train_set, test_set, dev_set = partition_data(data_orig_data, 0.9, 0.1)
        self.assertEqual(train_set.shape[0], 9)
        self.assertEqual(test_set.shape[0], 1)
        self.assertEqual(dev_set.shape[0], 0)

    def test_save_data(self):
        """
        Tests the save_data funciton. Creates test file TEST_save_result.txt and verfies contents
        """
        filename = 'TEST_save_result.txt'

        header = np.array(['Item1', 'Item2', 'Item3'])
        data = np.array([[7, 4, 1], [8, 5, 2]])

        expected_contents = 'Item1,Item2,Item3\n7,4,1\n8,5,2\n'

        save_data(filename, header, data)

        file_contents = ''
        with open(filename, 'r') as myfile:
            file_contents = myfile.read()

        self.assertEqual(file_contents, expected_contents)
