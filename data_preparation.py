"""
Data preparation and manipluation functions
"""
import os
import numpy as np
import pandas as pd


def load_data(filename, encode_list, column_drop_list):
    """
    Loads the data from a csv file
    This method will produce two datasets:

    The original data containing the data from the csv file

    A 'massaged data set' that will contain the data after two operaitons take place:
    1. drop of irrelevant columns as defined in the column_drop_list
    2. encoding of labelled values to numeric counterparts as defined in encode_list

    An analytics dictionary is also provided, with shape, description and correlation operaitons performed on the massaged dataset

    Arguments:
    encode_list -- list of the columns to encode to numerical values from the original dataset in the 'massaged dataset'
    column_drop_list -- list of the columns to drop from the original dataset in the 'massaged dataset'

    Returns:
    header orig -- the header row of the file containing the columns titles
    data_orig -- the data rows of the file
    header_massaged -- header row of the massaged dataset
    data_massaged -- data rows of the massaged dataset
    data_analytics -- dictionary of initial analysis of the data:
                            nulls -- flag indicationg theat the dataset conatins nulls
                            shape -- shape structure
                            description -- pandas anlysis
                            correlation -- correlation matrix
    """

    script_dir = os.path.dirname(__file__)
    abs_data_file_path = os.path.join(script_dir, filename)

    data_df = pd.read_table(abs_data_file_path, sep=',')

    contains_nulls = False
    # check that dataframe does not contain null values
    if data_df.isnull().values.any():
        contains_nulls = True

    header_orig = np.array(data_df.columns.values).squeeze()
    data_orig = np.array(data_df.values)

    # create a clone dataframe that will be 'massaged'
    data_df_copy = data_df.copy(deep=True)

    for encode_item in encode_list:
        data_df_copy[encode_item].replace(data_df_copy[encode_item].unique(), range(
            0, len(data_df_copy[encode_item].unique())), inplace=True)

    for drop_column_item in column_drop_list:
        del data_df_copy[drop_column_item]

    header_massaged = np.array(data_df_copy.columns.values).squeeze()
    data_massaged = np.array(data_df_copy.values)

    data_analytics = {
        'nulls' : contains_nulls,
        'shape': data_df_copy.shape,
        'description': data_df_copy.describe(),
        'correlation': data_df_copy.corr()
    }

    return header_orig, data_orig, header_massaged, data_massaged, data_analytics

def partition_data(data_orig, train_percentage=0.6, test_percentage=0.2):
    """
    Divides a dataset in three parts after randomly rearranging its rows. Default partitions
    are as follows:
    Training set -- 60% of data
    Testing set -- 20% of data
    Dev Set -- 20% of data

    Arguments:
    data_orig -- the original, complete dataset
    train_percentage -- the percentage of the dataset that will become training data
    test_percenntage -- the percentage of the dataset that will become testing data
                        hence the development dataset will become the remainder
    Returns:
    train_set -- the training data set
    test_set -- the testing data set
    dev_set -- the development data set
    """
    # shuffle data. as data is shuffled accross rows, it requires the double transpose
    np.random.shuffle(data_orig)
    data_orig = data_orig

    train_set_size = (int)(data_orig.shape[0] * train_percentage)
    test_set_size = (int)(data_orig.shape[0] * test_percentage)

    train_set = data_orig[:train_set_size, :]
    test_set = data_orig[train_set_size:train_set_size+test_set_size, :]
    dev_set = data_orig[train_set_size+test_set_size:, :]

    return train_set, test_set, dev_set

def save_data(filename, header, data, override=False):
    """
    saves a dataset to file

    Arguments:
    filename -- the name of the dataset file
    header -- the data header
    data -- the data information

    Returns:
    """

    script_dir = os.path.dirname(__file__)
    abs_data_file_path = os.path.join(script_dir, filename)

    if (os.path.exists(abs_data_file_path) is False) or (override is True):
        # create dataframe with right format
        data_frame = pd.DataFrame(data=data, columns=header)
        # save data, remove index coluum
        data_frame.to_csv(abs_data_file_path, index=False)
