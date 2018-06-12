# -*- coding: utf-8 -*-
import logging
import pandas as pd
import os

PARENT_FOLDER = os.path.abspath(os.path.join(__file__ ,"../../.."))
INPUT_PATH = os.path.join(PARENT_FOLDER,'data/raw/train.csv')
SUBMISSION_PATH = os.path.join(PARENT_FOLDER,'data/raw/submission_format.csv')
OUTPUT_PATH = os.path.join(PARENT_FOLDER,'data/processed/')

def meter_data_to_csv(train,meter_id):
    '''
    Save meter data to csv
    :param train:
    :param meter_id:
    :return:
    '''
    tmp = train.loc[train.meter_id == meter_id]
    submission = pd.read_csv(SUBMISSION_PATH)
    if meter_id.startswith('38'):
        meter = pd.merge(submission.loc[submission.meter_id == '38_9686'], tmp[['Timestamp','Values']], how='left',on=['Timestamp'])
    else:
        meter = pd.merge(submission.loc[submission.meter_id == meter_id],tmp,how='left',on=['meter_id','Timestamp'])
    meter.to_csv(OUTPUT_PATH+meter_id+'.csv',index=False)

def main():
    ''' Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #read train data
    train = pd.read_csv(INPUT_PATH).iloc[:,1:]

    # get data of meter 234_203
    meter_id1 = '234_203'
    meter_data_to_csv(train,meter_id1)

    # get data of meter 234_203
    meter_id2 = '334_61'
    meter_data_to_csv(train,meter_id2)

    # get data of meter3, a little special, use demand meter and reactive meter
    meter_demand = '38_9687'
    meter_data_to_csv(train,meter_demand)
    meter_reactive = '38_9688'
    meter_data_to_csv(train,meter_reactive)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
