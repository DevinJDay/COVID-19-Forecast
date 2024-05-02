import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px
from collections import namedtuple
import os
from distributed import Client
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics

df = pd.read_csv('/Users/daijie/Desktop/MachineLearning/Ramin UCLA/COVID19/dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

confirmed_df = df[['Date', 'Country', 'Confirmed']]
all_countries = confirmed_df['Country'].unique()
deaths_df = df[['Date','Country', 'Deaths']]
recovered_df = df[['Date','Country','Recovered']]

days_to_forecast = 90
all_pms = []

mode = 'custom'
for country in all_countries: 

    try:
        assert(country in confirmed_df['Country'].values)
        print('Country/Region ' + str(country) + ' is listed! ')
        country_confirmed_df = confirmed_df[(confirmed_df['Country'] == country)]
        country_deaths_df = deaths_df[(deaths_df['Country'] == country)]
        country_recovered_df = recovered_df[(recovered_df['Country'] == country)]
        country_dfs = [('Confirmed', country_confirmed_df), ('Deaths', country_deaths_df), ('Recovered', country_recovered_df)]
        for country_df_tup in country_dfs:
          try:

              case_type = country_df_tup[0]
              country_df = country_df_tup[1]
              country_df = country_df[['Date', case_type]]
              country_df.columns = ['ds','y']
              country_df['ds'] = pd.to_datetime(country_df['ds'])
              country_df = country_df.groupby('ds').sum()[['y']].reset_index()
        
      
              z = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
              z.add_seasonality(name='monthly', period=30.5, fourier_order=10)
              z.add_seasonality(name='weekly', period=7, fourier_order=21)
              z.add_seasonality(name='daily', period=1, fourier_order=3)
                
              #fit
              z.fit(country_df)
              future = z.make_future_dataframe(periods=days_to_forecast)
              forecast = z.predict(future)
              
              #visualization and saving images
              save_path = '/Users/daijie/Desktop/MachineLearning/Ramin UCLA/COVID19/countries/'+mode+'/'+country +'/'+case_type+'/'
              forecast_plot = z.plot(forecast)
              ax = forecast_plot.gca()
              title = 'Forecasting'+' '+case_type+' '+'Cases of'+' '+country+' '+'with Prophet'
              ax.set_title(title,size = 24)
              forecast_plot.savefig(save_path+title+'.png')    
              ponents = z.plot_components(forecast)
              ponents.savefig(save_path+case_type+'_ponents.png')
              
              #cross validation
              z_cv = cross_validation(z, initial='366 days', period='15 days', horizon = '30 days',parallel="processes")
              pm = performance_metrics(z_cv, rolling_window=0)
              pm.to_csv(save_path+case_type+'_pm.csv')
              all_pm = performance_metrics(z_cv, rolling_window=1)
              all_pm['Country']=country
              all_pm['Casetype']=case_type 
              all_pms.append(all_pm)
                
              
              #visualization and saving images
              mape_plot = plot_cross_validation_metric(z_cv, metric='mape', rolling_window=0)
              mape_plot.savefig(save_path+case_type+'_mape.png')
              cross_validation_plot = z.plot(z_cv)
              ax = cross_validation_plot.gca()
              ax.set_title('Cross Validation, period=15 days, horizon = 30 days',size = 24)
              cross_validation_plot.savefig(save_path+case_type+'_cv.png')
          except:
              print('Oops! Someting went wrong!')
              continue

    except:
        print('Country '+country+' '+'is not listed!')
        continue

result = pd.concat(all_pms)
result.to_csv('/Users/daijie/Desktop/MachineLearning/Ramin UCLA/COVID19/countries/custom/all_pms.csv')

    
    