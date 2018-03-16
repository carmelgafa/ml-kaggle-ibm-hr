import constants
from analyse import analyse_attrition
from analyse import analysis_comparison_features
from analyse import analyse_data_init
from analyse import analyse_comparison

from analyse import plot_comparison_curve
from data_preparation import load_data

column_drop_list = [constants.EMPLYEENO_R, constants.EMPLOYEECOUNT_R,
                    constants.ISOVER18_R, constants.STDHOURS_R]
encode_list = [constants.GENDER_T, constants.STATUS_T, constants.DEPARTMENT_T, constants.ROLE_T,
               constants.OVERTIME_T, constants.TRAVEL_T, constants.ISRESIGNED_T, constants.EDUCATION_T]

file_path = '.\\datasets\\WA_Fn-UseC_-HR-Employee-Attrition.csv'

header, data, m_header, m_data, analytics = load_data(
    file_path, encode_list, column_drop_list)

#analyse_data_init(header, analytics)

features_to_analyse = [constants.STOCKOPTIONS, constants.TRAINING,
                        constants.SATISFACTION, constants.TEAMCLICK,
                        constants.ROLE_T, constants.LEVEL,
                        constants.DEPARTMENT_T,
                        constants.EDUCATION_T, constants.GENDER_T,
                        constants.COMPANIES,
                        constants.STATUS_T, constants.RATING,
                        constants.LIFEBALANCE, constants.INVOLVEMENT
                        ]

features_to_analyse = [constants.RATING,]

analyse_attrition(header, data, features_to_analyse, 'Yes', 'No')

comparison_sets = [(constants.SALARY, constants.AGE),
                    (constants.SALARY, constants.YEARSEMPLOYED),
                    (constants.LASTINCREMENTPERCENT, constants.RATING)]

analyse_comparison(m_header, m_data, comparison_sets)

#analyse_salary_attrition()
#def analyse_salary_attrition():
#    header, data, _, _,  _ = load_data('.\\datasets\\hr_train_1.csv', [], [])
#    analysis_comparison_features(header, data, constants.SALARY, constants.SATISFACTION,
#                                 constants.LIFEBALANCE, constants.DEPARTMENT_T, 'Human Resources', constants.TEAMCLICK)
