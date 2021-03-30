import numpy as np
import pandas as pd

def is_valid_category(key, value, dtype_dict):
    '''
    Check whether value in key:value pair is a valid categorical level (np.nan is also valid)
    Assumes that the input key exists in dtype_dict.
    Returns boolean
    '''
    try:
        if (value not in dtype_dict[key].categories) \
        and (not pd.isnull(value)):
            return(False)
        else: 
            return(True)
    except KeyError:
        raise Exception()


def obs_check(obs, dtype_dict):
    '''
    If sample observation contains invalid categorical levels,
    coerce the field(s) to the level `Other` (when possible)
    Parameters:
        obs (dict): sample observation in dictionary form
        dtype_dict (dict): dictionary denoting all column dtypes and categorical levels
    Returns:
        obs_new (dict): same sample observation, 
                        with any invalid categorical inputs forced to `Other`
    '''
    obs_new = obs
    for i in obs.keys():
        if (dtype_dict[i].type == pd.core.dtypes.dtypes.CategoricalDtypeType):
            if not is_valid_category(i,obs[i],dtype_dict):
                print('invalid categorical level: ',obs[i],'in field: ',i)
                obs_new[i] = 'Other'
                print("replaced with: 'Other' ")
        else: continue
    return(obs_new)



def dict_to_dataframe(input_dict, dtype_dict):
    ''' Convert input observation to dataframe, then set dtypes for each resulting column'''
    df = pd.DataFrame([input_dict]).astype(dtype_dict)
    return(df)



### The following functions are not used in the current build, 
### but may be implemented in the future.

def revenue_to_binary(revenue):
    ''' Convert revenue amount to a binary response variable '''
    return(revenue > 0)
    
    
def get_top_n_features(model, names, n):
    '''print the top (n) features of the model'''
    top_feature_df = pd.DataFrame(
        {'Feature': names,
         'Importance': model.feature_importances_
        }
    )
    top_feature_df.sort_values(by='Importance', ascending = False, inplace=True)
    return(top_feature_df.head(n=n))

