"""
Constants required for this excercise
items postfixed by _T are categorized and will need to be encoded
items postfixed by _R are redundant
"""

ISRESIGNED_T = 'Attrition'

# employee related information
AGE = 'Age'
EDUCATION_T = 'EducationField'
GENDER_T = 'Gender'
COMPANIES = 'NumCompaniesWorked'
STATUS_T = 'MaritalStatus'
HOMEDISTANCE = 'DistanceFromHome'

# role of the employee in the company
ROLE_T = 'JobRole'
LEVEL = 'JobLevel'
DEPARTMENT_T = 'Department'
YEARSCOMPANY = 'YearsAtCompany'
YEARSEMPLOYED = 'TotalWorkingYears'
YEARSROLE = 'YearsInCurrentRole'
YEARSLASTPROMO = 'YearsSinceLastPromotion'
YEARSMANAGER = 'YearsWithCurrManager'
TRAINING = 'TrainingTimesLastYear'

# satisfaction informaiton
SATISFACTION = 'JobSatisfaction'
TEAMCLICK = 'RelationshipSatisfaction'
LIFEBALANCE = 'WorkLifeBalance'
ENVIRONMENT = 'EnvironmentSatisfaction'

# salary and money related
SALARY = 'MonthlyIncome'
MONTHLYRATE = 'MonthlyRate'
DAILYRATE = 'DailyRate'  # Daily rate = the amount of money you are paid per day
HOURLYRATE = 'HourlyRate'
LASTINCREMENTPERCENT = 'PercentSalaryHike'# Percent salary hike = the % change in salary from 2016 vs 2015.
STOCKOPTIONS = 'StockOptionLevel' # Stock option level = how much company stocks you own.

# rating and involvement related
RATING = 'PerformanceRating'
INVOLVEMENT = 'JobInvolvement'
OVERTIME_T = 'OverTime'  # Y/N
TRAVEL_T = 'BusinessTravel'  # rare / requent

# redundant fields
EMPLYEENO_R = 'EmployeeNumber'  # number
EMPLOYEECOUNT_R = 'EmployeeCount'  # all 1
ISOVER18_R = 'Over18'  # all Y
STDHOURS_R = 'StandardHours'  # all 40

orig_file = '.\\datasets\\WA_Fn-UseC_-HR-Employee-Attrition.csv'
processed_file = '.\\datasets\\HR_attrition_orig_proc.csv'

train_x_file = '.\\datasets\\HR_attrition_train_x.csv'
test_x_file = '.\\datasets\\HR_attrition_test_x.csv'
train_y_file = '.\\datasets\\HR_attrition_train_y.csv'
test_y_file = '.\\datasets\\HR_attrition_test_y.csv'

train_x_nb_file = '.\\datasets\\HR_attrition_train_x_nb.csv'
test_x_nb_file = '.\\datasets\\HR_attrition_test_x_nb.csv'

train_x_lr_file = '.\\datasets\\HR_attrition_train_x_lr.csv'
test_x_lr_file = '.\\datasets\\HR_attrition_test_x_lr.csv'
train_x_nn_file = '.\\datasets\\HR_attrition_train_x_nn.csv'
test_x_nn_file = '.\\datasets\\HR_attrition_test_x_nn.csv'
