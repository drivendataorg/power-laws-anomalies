# -*- coding: utf-8 -*-
import pandas as pd
import os
import xgboost as xgb
from datetime import datetime,timedelta

PARENT_FOLDER = os.path.abspath(os.path.join(__file__ ,"../../.."))
DATA_PATH = os.path.join(PARENT_FOLDER,'data/processed/')
INPUT_PATH = os.path.join(PARENT_FOLDER,'data/raw/')
RESULT_PATH = os.path.join(PARENT_FOLDER,'data/result/')

def read_data(path_name):
    '''
    Convert data to timestamp indexed dataframe
    :param path_name:
    :return:converted dataframe
    '''
    dateparse = lambda date: pd.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(path_name, parse_dates=['Timestamp'], index_col='Timestamp', date_parser=dateparse)
    return df

def extract_features(df,weather):
    '''
    Extract features of input dataframe
    :param df: dataframe of current meter
    :param weather: weather of current meter
    :return: dataframe with extracted features
    '''
    df['month'] = df.index.month
    df['week'] = df.index.week
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['dayofweek'] = df.index.weekday
    df['temperature'] = weather.loc[df.index].fillna(method='ffill')
    df['is_off'] = df.index.weekday > 4
    df['is_pre_off'] = (df.index.weekday == 0) |  (df.index.weekday == 6)
    df['is_next_off'] = (df.index.weekday == 4) | (df.index.weekday == 5)
    return df

def handle_holidays(df,holidays):
    for date in holidays['Date']:
        df.loc[date, 'is_off'] = True
        pre_date = str(datetime.strptime(date, '%Y-%m-%d').date() - timedelta(days=1))
        next_date = str(datetime.strptime(date, '%Y-%m-%d').date() + timedelta(days=1))
        df.loc[pre_date, 'is_next_off'] = True
        df.loc[next_date, 'is_pre_off'] = True
    df = df.dropna()
    df['is_off'] = df['is_off'].astype('bool')
    df['is_pre_off'] = df['is_pre_off'].astype('bool')
    df['is_next_off'] = df['is_next_off'].astype('bool')
    df.index = pd.to_datetime(df.index)
    return df


def xgb_model(df,n_fold):
    for i in range(n_fold):
        test_idx = list(range(int(len(df) * i / n_fold), int(len(df) * (i + 1) / n_fold)))
        train_idx = [x for x in range(len(df)) if x not in test_idx]
        features = ['month', 'week', 'hour', 'dayofyear',
                    'dayofmonth', 'dayofweek', 'temperature', 'is_off', 'is_pre_off',
                    'is_next_off']
        X_train = df.loc[:, features].iloc[train_idx, :]
        y_train = df.loc[:, 'Values'].iloc[train_idx]
        X_test = df.loc[:, features].iloc[test_idx, :]
        y_test = df.loc[:, 'Values'].iloc[test_idx]
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        # model training
        params = {'seed': 33,
                  'objective': 'reg:linear',
                  'silent': 0,
                  'nthread': 1,
                  'max_depth': 6,
                  'learning_rate': 0.3}
        num_round = 5000
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=100)
        df['pred' + str(i)] = bst.predict(xgb.DMatrix(df.loc[:, features]))
    return df

def get_anomalies(df,n):
    pred_cols = [col for col in df if col.startswith('pred')]
    df['error'] = df['Values'] - df[pred_cols].mean(1)
    df_day = df.resample('D').mean()
    anomalies = df_day.loc[df_day['error'] > df_day['error'].mean() + n * df_day['error'].std()]
    # anomalies = anomalies.loc[df.range > df.range.quantile(0.75)]
    return anomalies.index.astype('str').tolist()

def pipline(df,weather,holidays,n_fold,n_sigma):
    df = df.resample('H').mean().dropna()
    weather = weather.groupby(weather.index)['Temperature'].mean()
    df = extract_features(df, weather)
    if holidays is not None:
        df = handle_holidays(df, holidays)
    df = xgb_model(df, n_fold)
    anomalies = get_anomalies(df, n_sigma)
    return anomalies

def mark_anomalies(df,anomalies):
    '''
    mark abnormal dates of raw data
    :param df:
    :param anomalies:
    :return: df with abnormal result
    '''
    df['is_abnormal'] = False
    for date in anomalies:
        df.loc[date,'is_abnormal'] = True
    return df.reset_index()[['obs_id', 'meter_id','Timestamp','is_abnormal']]


if __name__=='__main__':
    #weather & holidays
    weather = read_data(INPUT_PATH+'weather.csv')
    holidays = pd.read_csv(INPUT_PATH+'holidays.csv').iloc[:,1:]

    # meter1, 234_203
    meter_id1 = '234_203'
    print('Detecting meter1 weekday anomalies...')
    df1 = read_data(DATA_PATH + meter_id1 + '.csv')
    weather_site1 = weather[(weather.site_id == meter_id1) & (weather['Distance'] > 10) & (weather['Distance'] < 13)].iloc[:,
    1:].fillna(method='ffill')[['Temperature']]
    # meter1 is a combination of two meters as we described in the report,
    # here we only cares the first part because it has more abnormal patterns with higher confidence
    anomalies1 = pipline(df1[:'2015-06-20'],weather_site1,None,3,2.3)
    res1 = mark_anomalies(df1,anomalies1)

    # meter2, 334_61
    meter_id2 = '334_61'
    print('Detecting meter2 weekday anomalies...')
    df2 = read_data(DATA_PATH + meter_id2 + '.csv')
    weather_site2 = weather[(weather.site_id == meter_id2) & (weather['Distance'] >= 19)].iloc[:, 1:].fillna(method='ffill')[
        ['Temperature']]
    holidays_site2 = holidays[holidays.site_id == meter_id2]
    anomalies2 = pipline(df2,weather_site2,holidays_site2,4,3.5)
    res2 = mark_anomalies(df2,anomalies2)

    # meter3, 38_9687
    meter_id3 = '38_9687'
    print('Detecting meter3 weekday anomalies...')
    df3 = read_data(DATA_PATH + meter_id3 + '.csv')
    weather_site3 = weather[weather.site_id == '38'].iloc[:, 1:].fillna(method='ffill')[['Temperature']]
    holidays_site3 = holidays[holidays.site_id == '038']
    # separate into two parts as there exists an obvious turning point on reactive meter curve
    anomalies3_part1 = pipline(df3[:'2012-12-21'],weather_site3,holidays_site3,3,3.5)
    anomalies3_part2 = pipline(df3['2012-12-21':],weather_site3,holidays_site3,4,3.5)
    anomalies3 = anomalies3_part1 + anomalies3_part2
    res3 = mark_anomalies(df3,anomalies3)

    # combine results of three meters
    print('Combining weekday result...')
    res = res1.append(res2).append(res3)
    res.to_csv(RESULT_PATH+'res_weekday_p.csv',index=False)



