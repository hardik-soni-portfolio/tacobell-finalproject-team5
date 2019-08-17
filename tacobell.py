#import libraries
from flask import Flask,render_template,url_for,request
import pickle
import urllib.request
from sklearn.externals import joblib
import json
import datetime
import pandas as pd
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


offer_name_dict = {'ae264e3637204a6fb9bb56bc8210ddfd': 'bogo1',
    '4d5c57ea9a6940dd891ad53e9dbe8da0': 'bogo2',
    '3f207df678b143eea3cee63160fa8bed': 'informational1',
    '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'bogo3',
    '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'discount1',
    '2298d6c36e964ae4a3e7e9706d1fb8c2': 'discount2',
    'fafdcd668e3743c1bb461111dcafc2a4': 'discount3',
    '5a8bc65990b245e5a138643cd4eb9837': 'informational2',
    'f19421c1d4aa40978ebb69ca19b0e20d': 'bogo4',
    '2906b810c7d4411798c6938adc9daaa5': 'discount4'}

def user_user_recs(user_id, recom_user, offer_or_not, n_user = 10, top_n = 4):
    '''
    INPUT:
    user_id - an id of a user
    recom_user - user-promotion matrix
    offer_or_not - a pandas dataframe consisting of user_id and trans_not_from_offer_ratio
    n_user - number of similar users we want to use to create recommendation
    top_n - number of recommendation to be given to the user_id
    
    OUTPUT:
    the offer id of recommendation (list)
    '''
    recommendation = []
    if user_id in offer_or_not['user_id'].tolist():
        return 'do not offer anything'
    else:
        if user_id in recom_user.index:
            #find top 10 similar users 
            sim_users = []
            sim_users = find_similar_users(user_id)[:n_user]
            sim_users_df = pd.DataFrame()
            sim_users_df = pd.DataFrame(columns = recom_user.columns)
            for user in sim_users:
                user = user - 1
                sim_users_df = sim_users_df.append(recom_user.loc[user])
            #print(sim_users_df)
            recommendation = sim_users_df.sum().nlargest(top_n).index.tolist()
            recommendation.remove('user_id')
        else:
            recommendation = recom_user.sum().nlargest(top_n).index.tolist()
            recommendation.remove('user_id')
    return recommendation

def find_similar_users(user_id):
    '''
    INPUT:
    user_id - (int) a user_id
    recom_user - (pandas dataframe) matrix of users by prommotion
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first   
    '''
    recom_user = pd.read_csv('static/recom_user.csv', engine='c', index_col=0)
    # compute similarity of each user to the provided user
    user_row = recom_user.loc[user_id]
    #print(recom_user.shape)
    similarities = np.dot(user_row.T, recom_user.T)
    #print(similarities.shape)
    # create list of just the ids
    most_similar_users = list(((-similarities).argsort()) + 1)
    # remove the own user's id                             
    most_similar_users.remove(user_id)
    return most_similar_users # return a list of the users in order from most to least similar


def get_offer_id(offer_list):
    '''
    INPUT: a list of offer names (bogo1, discount1, etc)
    OUTPUT: a list of offer ids according to the portfolio dataframe
    '''
    offer_names = []
    reverse_offer_name = {v: k for k, v in offer_name_dict.items()}
    print(reverse_offer_name)
    for offer in offer_list:
        offer_names.append(reverse_offer_name[offer])
    return offer_names


def promotion_strategy(df,i):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    test = df
    loaded_model = i
    # Fit a model with treatment = 1 for all data points
    test['treatment'] = 1.0
    preds_treat = loaded_model.predict_proba(test, ntree_limit=loaded_model.best_ntree_limit)
    
    # Fit a model with treatment = 0 for all data points
    test['treatment'] = 0.0
    preds_cont = loaded_model.predict_proba(test, ntree_limit=loaded_model.best_ntree_limit)
    
    lift = preds_treat[:,1] - preds_cont[:,1]
    
    promotion = []
    
    for prob in lift:
        if prob > 0:
            promotion.append('Yes')
        else:
            promotion.append('No')

    promotion = np.array(promotion)
    
    return promotion


def score(df, promo_pred_col = 'offer_id',promo_pred_col_2 = 'quadrant'):
    n_treat       = df.loc[df[promo_pred_col] != 10 ,:].shape[0]
    n_control     = df.loc[df[promo_pred_col] == 10 ,:].shape[0]
    #n_treat_purch = df.loc[df[promo_pred_col] == 'Yes', 'purchase'].sum()
    n_treat_purch = df.loc[df[promo_pred_col_2] == 0 ].shape[0]
    
    #n_ctrl_purch  = df.loc[df[promo_pred_col] == 'No', 'purchase'].sum()
    n_ctrl_purch  = df.loc[df[promo_pred_col_2] == 1 ].shape[0]
    
    irr = n_treat_purch / n_treat - n_ctrl_purch / n_control
    nir = 10 * n_treat_purch - 0.15 * n_treat - 10 * n_ctrl_purch
    return (irr, nir)
 