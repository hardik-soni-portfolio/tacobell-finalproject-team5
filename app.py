#import all of the required packages and libraries
from flask import Flask, render_template, request
from datetime import datetime
import tacobell
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation')
def show_recommendation():
    return render_template('recommendation.html')

@app.route('/data', methods=['POST'])
def get_data():
    if request.method == 'POST':
        # Get user demographic inputs
        user_id = int(request.form['user_id'])

    offer_or_not = pd.read_csv('static/offer_or_not.csv',  engine='c')
    recom_user = pd.read_csv('static/recom_user.csv', engine='c')
    recommend_offer = tacobell.user_user_recs(user_id, recom_user, offer_or_not)

    recom_user_df = pd.read_csv('static/recommend.csv', engine='c')
    recom_user_list = list(recom_user_df['user_id'])
    recom_user_dict = {}
    for rec_usr in recom_user_list:
        rec_offer = tacobell.user_user_recs(rec_usr, recom_user, offer_or_not)
        recom_user_dict[rec_usr] = rec_offer
    final_recommendation = pd.DataFrame(list(recom_user_dict.items()), columns=['user_id', 'recommended_offer'])
    final_recommendation.to_csv('final_recommendation.csv')
    return render_template('results.html', recommended_offer = recommend_offer, recommend_xusers = final_recommendation)


@app.route('/edapromotion')
def show_edapromotion():
    return render_template('edapromotion.html')

@app.route('/edaforecast')
def show_edaforecast():
    return render_template('edaforecast.html')

@app.route('/forecast')
def show_forecast():
    return render_template('forecast.html')

@app.route('/revrescalc')
def show_rrcalc():
    test_data = pd.read_csv('static/rr_test_case.csv', engine='c')
    df = test_data[['day_num', 'per_id','age','became_member_on', 'income', 'gender_F', 'gender_M','gender_O']]

    discount1 = pickle.load(open("discount1.pickle.dat", "rb"))
    bogo2 = pickle.load(open("bogo2.pickle.dat", "rb"))
    bogo3 = pickle.load(open("bogo3.pickle.dat", "rb"))

    list_of_models = [discount1 , bogo2, bogo3]
    model_output_dict = {}
    for i in list_of_models:
        promos = tacobell.promotion_strategy(df,i)
        score_df = test_data.iloc[np.where(promos == 'Yes')]    
        irr, nir = tacobell.score(score_df)
        list1 = [round(irr,2), round(nir,2)]
        if i == discount1:
            key = "discount1"
        elif i == bogo2:
            key = "bogo2"
        elif i == bogo3:
            key = "bogo3"
        model_output_dict[key] = list1
    model_op_df = pd.DataFrame(list(model_output_dict.items()), columns=['Offer', 'IRR_and_NRR'])

    #print('Irr with this strategy is {:0.4f}.'.format(irr))
    #print('Nir with this strategy is {:0.2f}.'.format(nir))
    return render_template('revrescalc.html', model_op_df = model_op_df )

if __name__ == '__main__':
    app.run(debug=True)