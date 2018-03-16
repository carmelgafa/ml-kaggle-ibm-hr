from predict import digitize_columns
from predict import standardize_columns
from predict import encode_columns_nn

from data_preparation import load_data
from data_preparation import save_data

orig_file_path = '.\\datasets\\TESTING_ml.csv'
processed_file_path = '.\\datasets\\TESTING_ml_proc.csv'
nb_file_path = '.\\datasets\\TESTING_ml_nb.csv'
lr_file_path = '.\\datasets\\TESTING_ml_lr.csv'
nn_file_path = '.\\datasets\\TESTING_ml_nn.csv'

column_drop_list = ['Delete']
encode_list = ['Gender', 'Sport']
_, _, header, data, _ = load_data(orig_file_path, encode_list, column_drop_list)
save_data(processed_file_path, header, data)

column_bins_definition = {'Age':5, 'Salary':5}
data_nb = digitize_columns(header, data, column_bins_definition)
save_data(nb_file_path, header, data_nb)

columns_norm = ['Age', 'Salary']
data_lr = standardize_columns(header, data, column_bins_definition)
save_data(lr_file_path, header, data_lr)

print(header)
print(data_lr)
encode_list_nn = ['Gender', 'Sport']
header_nn, data_nn = encode_columns_nn(header, data_lr, encode_list_nn)
save_data(nn_file_path, header_nn, data_nn)
