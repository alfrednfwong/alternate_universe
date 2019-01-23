import pandas as pd
import numpy as np


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def get_day(datetime_object):
    '''This function returns only the date part of a datetime'''
    return datetime_object.date()

def get_hospital_data():
    '''
    Gets the Brazilian hospital no-show dataset
    :return:
        dataframe - features data
        series - target data
    '''
    df = pd.read_csv('./data/noshowappointments-kagglev2-may-2016.csv')
    df.columns = [
        'patient_ID', 'appointment_ID', 'gender', 'scheduled_day',
        'appointment_day', 'age', 'neighbourhood', 'scholarship',
        'hypertension', 'diabetes', 'alcoholism', 'handicap', 'SMS_received',
        'no_show'
    ]
    df.scheduled_day = pd.to_datetime(df.scheduled_day)
    df['scheduled_hour'] = (
        df.scheduled_day.apply(lambda x: x.hour)
    )
    # Parse the date from string
    df.appointment_day = pd.to_datetime(df.appointment_day)
    # Get lead_days column in timedelta64[ns] format
    df['lead_days'] = (
            df.appointment_day.apply(get_day)
            - df.scheduled_day.apply(get_day)
    )
    # Change the datatype into integer
    df.lead_days = (
        (df.lead_days.astype('timedelta64[D]')).astype(int)
    )
    # Create the appointment day-of-week column
    df['appointment_DOW'] = (
        df.appointment_day.dt.dayofweek
    )
    # Create lead_days_category column
    lead_days_labels = pd.Series([
        'A: Same day',
        'B: 1-2 days',
        'C: 3-7 days',
        'D: 8-31 days',
        'E: 32+ days'
    ])
    df['lead_days_category'] = pd.cut(
        df.lead_days, bins=[-1, 0, 2, 7, 31, 999],
        labels=lead_days_labels,
    )
    # Change dtype to string because when data in this column is put into the
    # comparison_df in the compare_by_column(df, column_name) function later on,
    # an 'overall' row can be appended to string values but not categorical values.
    df.lead_days_category = (
        df.lead_days_category.astype('str')
    )
    df.set_index('appointment_ID', inplace=True)
    df['is_female'] = (df.gender == 'F')
    df['age_group'] = (
        df.age.apply(lambda x: min(int(x / 10), 9))
    )
    columns_to_change = [
        'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'SMS_received'
    ]
    for column in columns_to_change:
        df[column] = (df[column] == 1)
    df['is_handicapped'] = (df.handicap > 0)
    boolean_replacement = {'Yes': True, 'No': False}
    df.no_show.replace(boolean_replacement, inplace=True)
    # create a new column no_show_last_time that takes the value of no_show in
    # the previous appointment of the same patient
    df = df.sort_values(
        by=['appointment_day', 'scheduled_day'], axis=0
    )
    df['no_show_last_time'] = (
        df.groupby('patient_ID')['no_show'].apply(lambda x: x.shift(1))
    )
    # Dropping the rows where the scheduled_day is later than the appointment_day
    df = (
        df.drop(df[df.lead_days < 0].index)
    )
    # change booleans into 1 and 0s
    bools = [
        'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'is_handicapped',
        'SMS_received', 'no_show', 'is_female', 'no_show_last_time'
    ]
    for elem in bools:
        df[elem] = df[elem] * 1
    # add dummy for NaN in no_show_last_time
    df['no_show_last_time_no_record'] = df.no_show_last_time.isna() * 1
    # change NaN values in no_show_last_time to 0, as we now have a dummy for NaNs
    df.no_show_last_time.fillna(0, inplace=True)
    # get dummy variables for the multiclass non-cardinal variables
    df = pd.get_dummies(
        df, dummy_na=False,
        columns=['scheduled_hour', 'appointment_DOW', 'neighbourhood'],
        drop_first=True
    )
    df = df.drop([
        'patient_ID', 'gender', 'scheduled_day', 'appointment_day', 'handicap',
        'lead_days_category', 'age_group'
    ], axis=1)

    # feature selection
    x_data = df.drop('no_show', axis=1)
    y_data = (df['no_show'])


    return x_data, y_data

def get_hospital_logistic_model(x_data, y_data):
    '''
    Returns the classifier and scaler of the hospital no show prediction

    :param: x_data. array-like (num_obs, num_features).
        Dataframe containing features
    :param: y_data. array-like (num_obs,). Target variable
    :return:
    classifier object
    scaler object
    '''

    y_data = np.array(y_data)
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    target_feature_name = 'no_show'
    RANDOM_STATE = 42

    lr = LogisticRegression(
        C=1000, random_state=RANDOM_STATE, solver='liblinear', max_iter=500,
        class_weight='balanced'
    )
    kf = KFold(n_splits=3, random_state=RANDOM_STATE, shuffle=True)

    for train_index, test_index in kf.split(x_data):
        x_train = x_data[train_index]
        y_train = y_data[train_index]
        x_test = x_data[test_index]
        y_test = y_data[test_index]

        lr.fit(x_train, y_train)
        train_pred = lr.predict(x_train)
        test_pred = lr.predict(x_test)
        train_score = roc_auc_score(y_train, train_pred)
        test_score = roc_auc_score(y_test, test_pred)

        print(f'train_score: {train_score}')
        print(f'test_score: {test_score}')

    return lr, scaler



