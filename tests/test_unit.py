import numpy as np
import pandas as pd

import pickle as pkl

# from rf_data_prep import CATEGORICAL_DTYPES

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

# unit test 1:
def test_is_valid_category():
    # testing with key:value parameters ('channelGrouping','Organic Search'):
    assert is_valid_category('channelGrouping','Organic Search',CATEGORICAL_DTYPES) == True

# unit test 2:
def test_obs_check():
    # testing whether function can successfully convert EDITED_OBS to match TRUE_OBS:
    assert obs_check(EDITED_OBS, CATEGORICAL_DTYPES) == TRUE_OBS

# unit test 3:
def test_dict_to_dataframe():
    # testing whether function can convert dictionary data input to Pandas.DataFrame object
    # with correct dtypes
    assert dict_to_dataframe(EDITED_OBS,CATEGORICAL_DTYPES).dtypes.to_dict() == CATEGORICAL_DTYPES

