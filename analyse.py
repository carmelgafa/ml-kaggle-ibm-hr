"""
kaggle HR analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import constants
from matplotlib.ticker import NullFormatter
from matplotlib.colors import ListedColormap

def analyse_data_init(header, analytics):
    """
    Displays initial analysis on a loaded dataset. In particular this funciton;

    1. indicates if null values are present
    2. prints the shape of the dataset
    3. prints the description information for the dataset, for each column
    4. plots the correlation matrix

    Arguments:

    header -- the data header for the loaded set
    analytics -- the analyics dictionary provided by the load_data function
    """

    if(analytics['nulls'] == True):
        print('Dataset contains null values')
    else:
        print('Dataset does not contain null values')

    print('Dataset shape: {}'.format(analytics['shape']))

    print(analytics['description'])

    plot_correlation_matrix(header, analytics['correlation'])

def plot_correlation_matrix(headers, correlations):
    """
    Plots the correlation matrix
    
    Arguments:
    headers -- the headers of the correleation graph
    correlations -- the correlation information
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations)
    fig.colorbar(cax)
    ticks = np.arange(0, len(headers), 1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(headers)
    ax.set_yticklabels(headers)

    for tick in ax.get_xticklabels():
        tick.set_rotation(80)

    plt.show()

def analyse_attrition(header, data, features_to_analyse, attrition_true, attrition_false):
    """
    In oder to print graphs tiled, create a figure at the beginng of the script and
    use the correct position when calling plot_attrition_curve; (e.g. 311, 312 313).
    Then call the plt.show at the end of the script.

    For seperate plots, create a figure for each plot, call plot_attrition_curve with
    position 111 and plt.show for each plot.

    Arguments:
    header -- the data header
    data -- the data
    """
    for feature in features_to_analyse:
        fig = plt.figure()
        results = analysis_attrition_feature(
            header, data, feature, constants.ISRESIGNED_T, attrition_true, attrition_false)
        plot_attrition_curve(results, feature, fig, 111)
        plt.show()

def analysis_attrition_feature(header, data, feature_id, attrition_feature_id,
                               attrition_true=1, attrition_false=0):
    """
    Analyses the attrition of a specified feature. For example if the feature
    is teh departments this fucntion will give the number of employees,
    number of terminations and attrition per department

    Arguments:
    header -- the header infomation of this dataset
    data -- the dato to process
    feature_id -- the column name of the feature to examine (e.g. Departments)
    left_feature_id -- the column name of the attrition column
    left_true -- the value of the left column when the employee has left
    left_false -- the value of the left column when the employee is still employed

    Returns:
    results -- hashmap of results having:
                Categories, array of categories of theis feature
                Current, array of current employees for each feature
                Left, array of left employees for each feature
                Attrition, array for the attrition for each feature
    """
    results = {}

    # the index in the data that points to the attrition
    left_idx = np.argwhere(header == attrition_feature_id).squeeze()

    # the index in the data that points to the department of the employee
    feature_idx = np.argwhere(header == feature_id).squeeze()

    # the names of the categories in this feature
    # e.g. all teh department names under the header departments
    categories = np.unique(data[:, np.argwhere(header == feature_id)])

    category_employees_current = np.empty((len(categories), 0), dtype=int)
    category_employees_left = np.empty((len(categories), 0), dtype=int)

    for category in categories:
        # filter the data for the feature
        feature_data = data[np.where(
            data[:, feature_idx] == category), :].squeeze()

        # extract emplyees still employed and add their count to the current employees array
        still_working = len(
            np.where(feature_data[:, left_idx] == attrition_false)[0])

        category_employees_current = np.append(
            category_employees_current, [still_working])

        # extract emplyees that left and add their count to the left employees array
        left = len(np.where(feature_data[:, left_idx] == attrition_true)[0])
        category_employees_left = np.append(category_employees_left, [left])
        #print('{} -{}'.format(feature, feature_employees_left.shape))

    feature_percentage_attrition = category_employees_left / (category_employees_current
                                                              + category_employees_left) * 100

    # move all results in hashmap
    results['categories'] = categories
    results['current'] = category_employees_current
    results['left'] = category_employees_left
    results['attrition'] = feature_percentage_attrition

    return results

def plot_attrition_curve(results, title, fig, position):
    """
    Plots the the number of employees employed vs the number of employees left
    of all the various categories for a given feature.
    For example if we cansider the departments feature, this function will plot a 
    bar graph of the current and resigned employees for eveery department.
    It will also superimpose the attrition values for each category

    Arguments:
    results -- dictionary containing all the data necessary
                'Features'
    """
    plt.gcf().subplots_adjust(bottom=0.4)

    categories = results['categories']
    employees_current = results['current']
    employees_left = results['left']
    percentage_attrition = results['attrition']

    X = range(0, len(categories), 1)

    ax1 = fig.add_subplot(position)
    fig.suptitle(title)

    ax1.bar(X, employees_current, color='b', alpha=0.5)
    ax1.bar(X, employees_left, color='r', alpha=0.5, bottom=employees_current)
    ax1.set_ylabel('Employees current & left')

    ax2 = ax1.twinx()
    ax2.plot(X, percentage_attrition, color='b')
    ax2.set_ylim([0, np.max(percentage_attrition)+1])
    ax2.set_ylabel('% Attrition')

    ax2.set_xticks(X)
    ax2.set_xticklabels(categories)

    for tick in ax1.get_xticklabels():
        tick.set_rotation(80)

def analyse_comparison(header, data, comparison_sets):
    for feature_1, feature_2 in comparison_sets:
        x, y, z = analysis_comparison_features(
            header, data, feature_1, feature_2, constants.ISRESIGNED_T)
        plot_comparison_curve(feature_1, feature_2, x, y, z)

def analysis_comparison_features(header, data, x_feature_id, y_feature_id, z_feature_id, filter_feature_id=None, filter_feature_value=None):
    """
    """
    # the index of the number of years at the company
    x_feature_idx = np.argwhere(header == x_feature_id).squeeze()

    # the index of the salary at the company
    y_feature_idx = np.argwhere(header == y_feature_id).squeeze()

    # the index of the salary at the company
    z_feature_idx = np.argwhere(header == z_feature_id).squeeze()

    if filter_feature_id != None:
        filter_feature_idx = np.argwhere(header == filter_feature_id)
        filtered_data = data[np.where(
            data[:, filter_feature_idx] == filter_feature_value), :]
    else:
        filtered_data = data

    x_data = filtered_data[:, x_feature_idx]
    y_data = filtered_data[:, y_feature_idx]
    z_data = filtered_data[:, z_feature_idx]

    return x_data, y_data, z_data

def plot_comparison_curve(x_title, y_title, x_data, y_data, z_data):

    left_color = 'lightcoral'
    remained_color = 'green'

    z_left = np.where(z_data == 0)
    z_remained = np.where(z_data == 1)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=(8, 8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    colors = [left_color, remained_color]

    axScatter.set_xlabel(x_title)
    axScatter.set_ylabel(y_title)

    axScatter.scatter(x_data, y_data, c=z_data, cmap=ListedColormap(colors), alpha=0.5)

    x_bins = min(30, len(np.unique(x_data)))

    axHistx.hist(x_data[z_left], bins=x_bins, color=left_color, alpha=0.5)
    axHistx.hist(x_data[z_remained], bins=x_bins, color=remained_color, alpha=0.5)

    x_left_hist = np.histogram(x_data[z_left], bins=x_bins)[0]
    x_remained_hist = np.histogram(x_data[z_remained], bins=x_bins)[0]
    # added a small factor to avoid division errors
    attrition = x_left_hist / (x_remained_hist + x_left_hist + 0.00005) * 100

    axHistx_2 = axHistx.twinx()
    axHistx_2.plot(np.histogram(x_data[z_left], bins=x_bins-1)[1], attrition, color='b')
    axHistx_2.set_ylim([0, 105])
    axHistx_2.set_ylabel('% Attrition')

    y_bins = min(30, len(np.unique(y_data)))

    axHisty.hist(y_data[z_left], bins=y_bins, color=left_color,
                 alpha=0.5, orientation='horizontal')
    axHisty.hist(y_data[z_remained], bins=y_bins, color=remained_color,
                 alpha=0.5, orientation='horizontal')

    plt.show()

def analysis_attrition_feature_drill(header, data, main_feature, secondary_feature):
    """
    """
    main_feature_idx = np.argwhere(header == main_feature).squeeze()

    main_features = np.unique(data[main_feature_idx])
    for main_feature in main_features:
        main_feature_data = data[:, np.where(
            data[main_feature_idx] == main_feature)[0]]

        fig = plt.figure()
        results = analysis_attrition_feature(
            header, main_feature_data, secondary_feature, constants.ISRESIGNED_T)
        plot_attrition_curve(results, main_feature+' - ' +
                             secondary_feature, fig, 111)
        plt.show()
