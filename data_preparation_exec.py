"""
This module is executed ont time and creates the test,
train and dev datasets to be used in this excercise
"""
from data_preparation import load_data
from data_preparation import partition_data
from data_preparation import save_data


def data_preparation(filename):
    """
    Executed only once to create train, test and dev datasets.

    Arguments:
    filename -- the file of the original data

    Returns:
    This function creates three files
    hr_train.csv -- the training dataset
    hr_test.csv -- the testing dataset
    hr_dev.csv -- the development dataset
    """

    # load originial dataset
    hr_header_orig, hr_data_orig, _, _, _ = load_data(filename, [], [])

    # partition data 60/20/20
    hr_train, hr_test, hr_dev = partition_data(hr_data_orig)

    # save results in files
    save_data('./datasets/hr_train.csv', hr_header_orig, hr_train)
    save_data('./datasets/hr_test.csv', hr_header_orig, hr_test)
    save_data('./datasets/hr_dev.csv', hr_header_orig, hr_dev)


data_preparation('./datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')
