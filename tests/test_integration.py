import numpy as np
import pandas as pd
import pickle as pkl

from rf_data_prep.read_observation import is_valid_category, obs_check, dict_to_dataframe

CATEGORICAL_DTYPES = pkl.load(open( "CATEGORICAL_DTYPES.pkl", "rb" ))

EDITED_OBS = {'channelGrouping': 'Organic Search',
 'visitNumber': 1,
 'device.browser': 'BAD DATA',
 'device.operatingSystem': 'BAD DATA',
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

TRUE_OBS = {'channelGrouping': 'Organic Search',
 'visitNumber': 1,
 'device.browser': 'Other',
 'device.operatingSystem': 'Other',
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

# integration test:
def test_integrated_read():
    ''' Integration Test
    Combining the following functions: 
        test_is_valid_category() -- determine whether value in key:value pair
                                    is a valid categorical level 
                test_obs_check() -- use test_is_valid_category() function to check
                                    single sample (dict) for any invalid categorical values.
                                    coerce any invalid field(s) to the level: `Other` 
        test_dict_to_dataframe() -- once sample is checked, convert to pandas DataFrame and
                                    set dtypes for each column
    '''
    my_sample = obs_check(EDITED_OBS, CATEGORICAL_DTYPES)
    my_sample_df = dict_to_dataframe(my_sample, CATEGORICAL_DTYPES)
    
    assert my_sample_df.dtypes.to_dict() == CATEGORICAL_DTYPES
