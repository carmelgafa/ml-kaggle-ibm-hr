"""
Analyse Testing
"""
import unittest
import numpy as np
import numpy.testing as nptest
import pandas as pd
from analyse import analysis_attrition_feature
from analyse import analysis_comparison_features


class TestAnalyse(unittest.TestCase):
    """
    Tests the functions in the analyse module
    """

    def test_analysis_attrition_feature(self):
        """
        Tests the analysis_attrition_feature function. Creates dataset with Dept
        and Left columns
        """
        header = np.array(['Dept', 'Left'])

        data = np.array(pd.DataFrame([['A', 0], ['A', 0], ['A', 0], ['A', 1],
                                      ['B', 0], ['B', 0], ['B', 1], ['B', 1],
                                      ['C', 0], ['C', 1], ['C', 1], ['C', 1]], index=None))

        results = analysis_attrition_feature(header, data, 'Dept', 'Left')

        nptest.assert_array_equal(results['Features'], ['A', 'B', 'C'])
        nptest.assert_array_equal(results['Current'], [3, 2, 1])
        nptest.assert_array_equal(results['Left'], [1, 2, 3])
        nptest.assert_array_equal(results['Attrition'], [25., 50., 75.])

    def test_analysis_comparison_features(self):
        """
        Tests analysis_comparison_features.
        Dataset used as follows
        -------------------------------------
        | X     | Y     | Z     | Filter    |
        -------------------------------------
        | 1     | 1     | 1     | 'Y'       |
        | 2     | 2     | 1     | 'Y'       |
        | 3     | 3     | 0     | 'Y'       |
        | 4     | 4     | 0     | 'N'       |
        -------------------------------------
        """
        header = np.array(['X', 'Y', 'Z', 'Filter'])

        data = np.array(pd.DataFrame([[1, 1, 1, 'Y'],
                                      [2, 2, 1, 'Y'],
                                      [3, 3, 0, 'Y'],
                                      [4, 4, 0, 'N']], index=None))

        x_data, y_data, z_data = analysis_comparison_features(
            header, data, 'X', 'Y', 'Z', 'Filter', 'Y')

        nptest.assert_array_equal(x_data, [1, 2, 3])
        nptest.assert_array_equal(y_data, [1, 2, 3])
        nptest.assert_array_equal(z_data, [1, 1, 0])

        x_data, y_data, z_data = analysis_comparison_features(
            header, data, 'X', 'Y', 'Z')

        nptest.assert_array_equal(x_data, [1, 2, 3, 4])
        nptest.assert_array_equal(y_data, [1, 2, 3, 4])
        nptest.assert_array_equal(z_data, [1, 1, 0, 0])
