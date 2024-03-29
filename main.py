import numpy as np
import pandas as pd

import boto3
import s3fs 

import pickle as pkl
import logging

from rf_data_prep.read_observation import is_valid_category, obs_check, dict_to_dataframe

SAMPLE_OBS = {'channelGrouping': 'Organic Search',
 'visitNumber': 1,
 'device.browser': 'Firefox',
 'device.operatingSystem': 'Macintosh',
 'device.isMobile': False,
 'device.deviceCategory': 'desktop',
 'geoNetwork.continent': 'Oceania',
 'geoNetwork.subContinent': 'Australasia',
 'geoNetwork.country': 'Australia',
 'geoNetwork.region': 'not available in demo dataset',
 'geoNetwork.metro': 'not available in demo dataset',
 'geoNetwork.city': 'not available in demo dataset',
 'totals.transactionRevenue': False,
 'trafficSource.campaign': '(not set)',
 'trafficSource.source': 'google',
 'trafficSource.medium': 'organic',
 'trafficSource.adwordsClickInfo.page': np.nan,
 'trafficSource.adwordsClickInfo.slot': np.nan,
 'trafficSource.adwordsClickInfo.adNetworkType': np.nan,
 'trafficSource.adContent': np.nan}


BUCKET_NAME = "stats404-mori"
s3 = boto3.resource('s3')

# CATEGORICAL_DTYPES = pkl.load(open( "CATEGORICAL_DTYPES.pkl", "rb" ))
CATEGORICAL_DTYPES = pkl.loads(s3.Bucket(BUCKET_NAME).Object("CATEGORICAL_DTYPES.pkl").get()['Body'].read())


logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    ### ---------------------------------------------------------------------------
    ### --- Part 1: Load Data to DataFrame
    ### ---------------------------------------------------------------------------
    LOGGER.info('Loading Input Data...')
    df = dict_to_dataframe(obs_check(SAMPLE_OBS,CATEGORICAL_DTYPES),CATEGORICAL_DTYPES)

    ### ---------------------------------------------------------------------------
    ### --- Part 2: Feature Engineering
    ### ---------------------------------------------------------------------------
    LOGGER.info('Peforming Feature Engineering on input data...')

    df['totals.transactionRevenue'] = df['totals.transactionRevenue'] > 0
    df.drop(['totals.transactionRevenue'], axis=1, inplace=True)
    df_encoded = pd.get_dummies(df, dummy_na=True)

    ### ---------------------------------------------------------------------------
    ### --- Part 3: Load Model and Run Prediction
    ### ---------------------------------------------------------------------------

    # model = pkl.load(open("rf_classifier_model.pkl", "rb" ))
    model = pkl.loads(s3.Bucket(BUCKET_NAME).Object("rf_classifier_model.pkl").get()['Body'].read())

    LOGGER.info('Model Loaded. Running Prediction...')

    try:
        prediction = model.predict(df_encoded)
        print({'Predicted to Make Purchase?' : bool(prediction)})
    except:
        LOGGER.info('Prediction Failed. Please check input data')
        print({'Predicted to Make Purchase?' : ''})





