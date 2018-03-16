import constants
import time
import numpy as np
import pandas as pd
import constants

from data_preparation import load_data
from data_preparation import save_data
from data_preparation import partition_data

 
def digitize_columns(header, data, column_bins_def):
 
    data_copy = np.copy(data)
 
    for column_id in column_bins_def:
        # get the column index
        column_idx = np.argwhere(header == column_id).squeeze()
        # column data
        column_data = data_copy[::, column_idx]
         
        bins = np.linspace(np.min(column_data), np.max(column_data), column_bins_def[column_id])
 
        column_data = np.digitize(column_data, bins)
        data_copy[:, column_idx] = column_data
 
    return data_copy
 
def standardize_columns(header, data, columns_norm):
 
    data_copy = np.array(data, dtype=float)

    for column_id in columns_norm:
        # get the column index
        column_idx = np.argwhere(header == column_id).squeeze()
         
        # column data
        column_data = data_copy[::, column_idx]
        mean = np.mean(column_data)
        std_dev = np.std(column_data)
        column_data = np.divide((column_data - mean), std_dev)

        data_copy[:, column_idx] = column_data

    return data_copy
 
def feature_scale_columns(header, data, columns_norm):

    data_copy = np.array(data, dtype=float)

    for column_id in columns_norm:
        # get the column index
        column_idx = np.argwhere(header == column_id).squeeze()
         
        # column data
        column_data = data_copy[::, column_idx]
        maximum = np.max(column_data)
        minimum = np.min(column_data)
        column_data = np.divide((column_data - minimum), (maximum - minimum))

        data_copy[:, column_idx] = column_data

    return data_copy


 
def encode_columns_nn(header, data, columns_norm):
 
    data_copy = np.copy(data, order='K').reshape(data.shape)
    header_copy = np.copy(header, order='K')
 
    for column_id in columns_norm:


        # get the column index
        column_idx = np.argwhere(header_copy == column_id).squeeze()
        # column data
        column_data = data_copy[::, column_idx]
        encoded_column_data = pd.get_dummies(column_data, prefix=column_id, sparse=False)
 
 
        data_copy = np.delete(data_copy, column_idx, axis=1)
        header_copy = np.delete(header_copy, column_idx)
 
        x = encoded_column_data.values
 
        header_copy = np.insert(header_copy, 0, encoded_column_data.columns.values)
        data_copy = np.insert(data_copy, 0, x.T, axis=1)
 
    return  header_copy, data_copy

def execute_classifier(classifier_name, classifier, train_x, train_y, test_x, test_y):
    '''
    Executes a classifier given the test and train data. Calculates the execution time for the 
    training

    Arguments:
    classifier_name -- the name of the classifier, for printing purposes
    classifier -- the actual classifier
    train_x -- training x data
    train_y -- training y data
    test_x -- testing x data
    test_y -- testing y data

    Returns:
    prints the result of the training and testing, together with the traioning execution time required
    test-score -- the testing score for this classifier
    '''
    t_start = time.clock()
    classifier.fit(train_x, train_y.ravel())
    t_end = time.clock()
    time_diff = t_end - t_start

    train_score = classifier.score(train_x, train_y)
    test_score = classifier.score(test_x, test_y)

    print('{} -  \t train score: {},\t test score: {},\t time:{}'.format(
        classifier_name, train_score, test_score, time_diff))

    return test_score

def prepare_model_data(rewrite_files=False):
    '''
    Prepares the data set files used by the models.

    Returns:
    results -- dictionary containing the following
        train_x_nb -- training x for naive bayes data set
        train_y -- training y data set
        test_x_nb -- test x for naive bayes data set
        test_y -- test y data set
        train_x_nn -- training x for neural nets (& Logistic regression)  data set
        test_x_nn -- test x for neural nets (& Logistic regression)  data set
    '''

    # step 1:
    column_drop_list = [constants.EMPLYEENO_R, constants.EMPLOYEECOUNT_R,
                        constants.ISOVER18_R, constants.STDHOURS_R]
    encode_list = [constants.GENDER_T, constants.STATUS_T, constants.DEPARTMENT_T, constants.ROLE_T,
                   constants.OVERTIME_T, constants.TRAVEL_T, constants.ISRESIGNED_T, constants.EDUCATION_T]

    _, _, m_header, m_data, _ = load_data(
        constants.orig_file, encode_list, column_drop_list)

    save_data(constants.processed_file, m_header, m_data, override=rewrite_files)

    # step 2:
    train_data, test_data, _ = partition_data(m_data, 0.8, 0.2)

    output_idx = np.argwhere(m_header == constants.ISRESIGNED_T).squeeze()

    train_y = train_data[:, output_idx]
    train_x = np.delete(train_data, output_idx, 1)

    test_y = test_data[:, output_idx]
    test_x = np.delete(test_data, output_idx, 1)

    traintest_header = np.delete(m_header, output_idx, 0)

    save_data(constants.train_x_file, traintest_header, train_x, override=rewrite_files)
    save_data(constants.train_y_file, np.array([constants.ISRESIGNED_T]), train_y, override=rewrite_files)
    save_data(constants.test_x_file, traintest_header, test_x, override=rewrite_files)
    save_data(constants.test_y_file, np.array([constants.ISRESIGNED_T]), test_y, override=rewrite_files)

    # to execute naive bayes we will discretise continuous data
    column_bins_definition = {constants.AGE: 10, constants.DAILYRATE: 10,
                              constants.HOMEDISTANCE: 10, constants.SALARY: 10,
                              constants.HOURLYRATE: 10, constants.MONTHLYRATE: 10,
                              constants.YEARSEMPLOYED: 5, constants.YEARSCOMPANY: 5,
                              constants.YEARSROLE: 5, constants.YEARSLASTPROMO: 5,
                              constants.YEARSMANAGER: 5, constants.LASTINCREMENTPERCENT: 16}

    train_x_nb = digitize_columns(
        traintest_header, train_x, column_bins_definition)
    test_x_nb = digitize_columns(
        traintest_header, test_x, column_bins_definition)

    save_data(constants.train_x_nb_file, traintest_header, train_x_nb, override=rewrite_files)
    save_data(constants.test_x_nb_file, traintest_header, test_x_nb, override=rewrite_files)

    columns_norm = [constants.AGE, constants.DAILYRATE,
                    constants.HOMEDISTANCE, constants.SALARY,
                    constants.HOURLYRATE, constants.MONTHLYRATE,
                    constants.YEARSEMPLOYED, constants.YEARSCOMPANY,
                    constants.YEARSROLE, constants.YEARSLASTPROMO,
                    constants.YEARSMANAGER, constants.LASTINCREMENTPERCENT]

    train_x_lr = feature_scale_columns(traintest_header, train_x, columns_norm)
    test_x_lr = feature_scale_columns(traintest_header, test_x, columns_norm)

    save_data(constants.train_x_lr_file, traintest_header, train_x_lr, override=rewrite_files)
    save_data(constants.test_x_lr_file, traintest_header, test_x_lr, override=rewrite_files)

    encode_list_nn = [constants.GENDER_T, constants.STATUS_T, constants.DEPARTMENT_T, constants.ROLE_T,
                      constants.OVERTIME_T, constants.TRAVEL_T, constants.EDUCATION_T,
                      constants.ENVIRONMENT, constants.INVOLVEMENT, constants.LEVEL,
                      constants.SATISFACTION, constants.COMPANIES, constants.RATING, constants.TEAMCLICK,
                      constants.STOCKOPTIONS, constants.TRAINING, constants.LIFEBALANCE]

    train_test_header_nn, train_x_nn = encode_columns_nn(
        traintest_header, train_x_lr, encode_list_nn)
    _, test_x_nn = encode_columns_nn(
        traintest_header, test_x_lr, encode_list_nn)

    save_data(constants.train_x_nn_file, train_test_header_nn, train_x_nn, override=rewrite_files)
    save_data(constants.test_x_nn_file, train_test_header_nn, test_x_nn, override=rewrite_files)

    results = {}

    results['train_x_nb'] = train_x_nb
    results['train_y'] = train_y
    results['test_x_nb'] = test_x_nb
    results['test_y'] = test_y
    results['train_x_nn'] = train_x_nn
    results['test_x_nn'] = test_x_nn
    return results

def load_model_data():
    '''
    Loads the model datasets from the respective files

    Returns:
    results -- dictionary containing the following
        train_x_nb -- training x for naive bayes data set
        train_y -- training y data set
        test_x_nb -- test x for naive bayes data set
        test_y -- test y data set
        train_x_nn -- training x for neural nets (& Logistic regression)  data set
        test_x_nn -- test x for neural nets (& Logistic regression)  data set
    '''
    _, train_x_nb, _, _, _ = load_data(constants.train_x_nb_file,[],[])
    _, train_y, _, _, _  = load_data(constants.train_y_file,[],[])
    _, test_x_nb, _, _, _ = load_data(constants.test_x_nb_file,[],[])
    _, test_y, _, _, _ = load_data(constants.test_y_file,[],[])
    _, train_x_nn, _, _, _ = load_data(constants.train_x_nn_file,[],[])
    _, test_x_nn, _, _, _ = load_data(constants.test_x_nn_file,[],[])

    results = {}
    results['train_x_nb'] = train_x_nb
    results['train_y'] = train_y
    results['test_x_nb'] = test_x_nb
    results['test_y'] = test_y
    results['train_x_nn'] = train_x_nn
    results['test_x_nn'] = test_x_nn

    return results

