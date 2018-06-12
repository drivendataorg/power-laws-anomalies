import os
import pandas as pd

PARENT_FOLDER = os.path.abspath(os.path.join(__file__ ,"../../.."))
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

def combine_results(df_weekend,df_weekday):
    '''
    Combine anomaly results of weekend and weekdays
    :param df_weekend:
    :param df_weekday:
    :return: combined dataframe
    '''
    tmp = df_weekday.loc[df_weekday.is_abnormal == True][['meter_id']].reset_index().drop_duplicates().set_index('Timestamp')
    tmp['date'] = tmp.index.date
    tmp.drop_duplicates(inplace=True)
    for index,row in tmp.iterrows():
        df_weekend.loc[(df_weekend.meter_id==row['meter_id'])&(df_weekend.index.date==row['date']),'is_abnormal'] = True
    return df_weekend.reset_index()[['obs_id', 'meter_id','Timestamp','is_abnormal']]

if __name__=='__main__':
    # read weekend and weekday anomaly results
    res_weekend = read_data(RESULT_PATH + 'res_weekend_with_holiday.csv')
    res_weekday_r = read_data(RESULT_PATH + 'res_weekday_r.csv')
    res_weekday_p = read_data(RESULT_PATH + 'res_weekday_p.csv')

    # get r version combined results
    res_combined_r = combine_results(res_weekend.copy(), res_weekday_r)
    res_combined_r.to_csv(RESULT_PATH + 'res_combined_r.csv', index=False)

    # get python version combined results
    res_combined_p = combine_results(res_weekend.copy(), res_weekday_p)
    res_combined_p.to_csv(RESULT_PATH + 'res_combined_p.csv', index=False)






