# -*- coding: utf-8 -*-
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from sklearn.cluster import KMeans 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pathlib

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Temperature_change_Data.csv")))

def process(df):
  df_c =df.copy()
  df_c.set_index("year", inplace=True)
  df_c = df_c.loc[[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]]
  df_c.reset_index(inplace = True)
  df_c = df_c.groupby(['Country Name',]).agg({'tem_change':'mean',})
  df_c.reset_index(inplace = True)
  return df_c

def most_graph():
    df_c = process(df)
    df_c = df_c.sort_values(by=['tem_change'],ascending=False).head(10)
    fig = px.bar(df_c, x="Country Name", y='tem_change' ,text='tem_change', title="Top ten countries that have highest temperature change in the last decades")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(autosize=True,  
        template='seaborn', paper_bgcolor="rgb(234, 234, 242)", legend=dict(orientation="v", yanchor="bottom",
            y=0.3, xanchor="left", x=1.02 ))
    fig.update_xaxes( tickangle = 10, title_text = "Countries", title_font = {"size": 15}, title_standoff = 0)
    fig.update_yaxes(showticklabels=False,tickmode="auto", title='Temperature Change',title_standoff = 0)
    return fig

def least_graph():
    df_c = process(df)
    df_c = df_c.sort_values(by=['tem_change'],ascending=True).head(10)
    fig = px.bar(df_c, x="Country Name", y='tem_change',text='tem_change' , title="Top ten countries that have lowest temperature change in the last decades")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(autosize=True,template='seaborn',
                paper_bgcolor="rgb(234, 234, 242)",legend=dict( orientation="v",yanchor="bottom",y=0.3,xanchor="left",x=1.02))
    fig.update_xaxes( tickangle = 10,title_text = "Countries",title_font = {"size": 15},title_standoff = 0)
    fig.update_yaxes(showticklabels=False, title='Temperature Change')
    return fig


def lregg(df, selected_con, selected_mon):
    df_lr = df.copy()
    df_lr.dropna(axis=0)
    df_lr = df_lr.drop(['Country Code'], axis=1)
    df_lr['year'] = df_lr['year'].astype('int32')
    df_lr = df_lr[df_lr['Months'] == selected_mon]
    df_lr = df_lr[df_lr['Country Name'] == selected_con]
    x= np.array(df_lr['year'].values.tolist())
    y= np.array(df_lr['tem_change'].values.tolist())
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    lr1 = LinearRegression()
    lr1.fit(x_train, y_train)
    return lr1, x_test, y_test

def code():
    df1 = df.copy()
    df1.dropna(axis=0)
    df1 = df1.drop(['Country Code'], axis=1)
    le1 = preprocessing.LabelEncoder()
    df1['Country Name']= le1.fit_transform(df1['Country Name'])
    le2 = preprocessing.LabelEncoder()
    df1['Months']= le2.fit_transform(df1['Months'])
    df1['year'] = df1['year'].astype('int32')
    df1 = df1[df1.tem_change.isnull() == False]
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
    df1['y_predict']= kmeans.fit_predict(df1) 
    df1['Country Name']= le1.inverse_transform(df1['Country Name'])
    df1['Months']= le2.inverse_transform(df1['Months'])
    df_countrycode=pd.read_csv(os.path.join(APP_PATH, os.path.join("data", 'FAOSTAT_data_11-24-2020.csv')))
    df_countrycode.drop(['Country Code','M49 Code','ISO2 Code','Start Year','End Year'],axis=1,inplace=True)
    df_countrycode.rename(columns = {'Country':'Country Name','ISO3 Code':'Country Code'},inplace=True)
    df3 = pd.merge(df1, df_countrycode, how='outer', on='Country Name')
    return df3

mon_dropdown = dcc.Dropdown(options=df['Months'].unique(),
                            value='Meteorological year')

con_dropdown = dcc.Dropdown(options=df['Country Name'].unique(),
                            value = 'World')


app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH],    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],)
app.title = "Temperature Change Dashboard"
server = app.server
app.config["suppress_callback_exceptions"] = True

app.layout = html.Div([
    html.H1('Temperature Change Dashboard',style={'padding': '3vh','padding-bottom': '0','margin-bottom': '0','text-align':'center'}),
    dbc.Table( html.Tr([
        html.Td(con_dropdown), html.Td(mon_dropdown),
        ]),style={'margin-bottom': '0'}),

    dcc.Tabs(style={'height': '5vh',}, children = [
        dcc.Tab(label='Maximum And Minimum', style={'padding': '0','line-height': '5vh'},
        selected_style={'padding': '0','line-height': '5vh'}, children=[
            dbc.Table( html.Tr([
                html.Td(dcc.Graph(id='least-graph', figure= least_graph(),style={'width': '49vw'})),
                html.Td(dcc.Graph(id='most-graph', figure= most_graph(),style={'width': '49vw'})),
            ])),
        ]),
        dcc.Tab(label='Seasonal And Monthly', style={'padding': '0','line-height': '5vh'},
        selected_style={'padding': '0','line-height': '5vh'}, children=[
            html.H3('Seasonal And Monthly Temperature Change',style={'padding': '2vh','padding-bottom': '0','margin-bottom': '0'}),
            dbc.Table( html.Tr([
                html.Td(dcc.Graph(id='season-graph',style={'width': '49vw'})),
                html.Td(dcc.Graph(id='month-graph',style={'width': '49vw'})),
            ])),
        ]),
        dcc.Tab(label='Future Prediction', style={'padding': '0','line-height': '5vh'},
        selected_style={'padding': '0','line-height': '5vh'}, children=[
            html.H3('Temperature Change Prediction Using Linear Regression',style={'padding': '2vh','padding-bottom': '0','margin-bottom': '0'}),
            dbc.Table( html.Tr([
                html.Td([
                    dbc.Table( html.Tr([ html.Td(html.H4('Enter Year')),
                                        html.Td(dcc.Input(id='input-1-state', type='text', value='2000'))
                                        ])),
                    html.Button(id='submit-button', n_clicks=0, children='Predict', style={'color':'primary', 'margin':'1vh'}),
                    html.H4(id='output-state')]),
                html.Td(dcc.Graph(id = 'reg-graph',style={'width': '49vw'})),
            ])),
        ]),
        dcc.Tab(label='Temperature Zones', style={'padding': '0','line-height': '5vh'},
        selected_style={'padding': '0','line-height': '5vh'}, children=[
            dbc.Table( html.Tr([
                html.Td([html.H4('Worldwide Temperature Change Trend',style={'margin-bottom': '0'}),
                    dcc.Graph(id='world-graph',style={'width': '49vw'})]),
                html.Td([html.H4('Countries with Similar Temperature Change Trend - KMeans',
                                    style={'margin-bottom': '0'}),
                    dcc.Graph(id='kworld-graph',style={'width': '49vw'})]),
            ])),
        ]),
    ])
], style={'height':'95vh'})


@callback(Output(component_id='season-graph', component_property='figure'),
              [Input(component_id=con_dropdown, component_property='value')])
def season_graph(selected_con):
    df0 = df[df['Country Name'] == selected_con]
    df1 = df0[df0['Months'] == 'Winter']
    df2 = df0[df0['Months'] == 'Spring']
    df3 = df0[df0['Months'] == 'Summer']
    df4 = df0[df0['Months'] == 'Fall']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df1['year'], y=df1.tem_change, mode='lines',name='Winter'))
    fig.add_trace(go.Scatter(x = df2['year'] , y=df2.tem_change,mode='markers',name='Spring'))
    fig.add_trace(go.Scatter(x = df3['year'] , y=df3.tem_change,mode='lines', name='Summer'))
    fig.add_trace(go.Scatter(x = df4['year'] , y=df4.tem_change,mode='markers', name='Fall'))
    fig.add_annotation(x='55',y=2.165,xref="x", yref="y",text="The hottest winter",showarrow=True,
            font=dict(family="Courier New, monospace",size=16,color="#ffffff"),
            align="center",arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="#636363",ax=20,ay=-30,
            bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ff7f0e",opacity=0.8)
    fig.update_layout(autosize=True,template='seaborn',
        paper_bgcolor="rgb(234, 234, 242)",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    fig.update_xaxes(type='category',title='Years')
    fig.update_yaxes(title='Temperature Change')
    return fig


@callback(Output(component_id='month-graph', component_property='figure'),
              [Input(component_id=con_dropdown, component_property='value')])
def month_graph(selected_con):
    df0 = df[df['Country Name'] == selected_con]
    df0.set_index("Months", inplace=True)
    df0 = df0.loc[['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December' ]]
    df0.reset_index(inplace = True)
    
    fig = px.line_polar(df0, r=df0.tem_change, theta=df0.Months,animation_frame='year', line_close=True)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[-0.5, 3])),autosize=True,template='seaborn', paper_bgcolor="rgb(234, 234, 242)",
    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig


@callback(Output(component_id='reg-graph', component_property='figure'),
              [Input(component_id=con_dropdown, component_property='value'),
              Input(component_id=mon_dropdown, component_property='value')])
def page_4_dropdown(selected_con, selected_mon): 
    lr1, x_test, y_test = lregg(df, selected_con, selected_mon)
    dfxy = pd.DataFrame(data = np.hstack((x_test, y_test.reshape(-1, 1))), columns = ['year', 'tem_change'])
    dfxy['y_pred_test'] = lr1.predict(x_test)
    fig = px.scatter(dfxy, x='year', y='tem_change', opacity=0.65)
    fig.add_traces(go.Scatter(x=dfxy.year, y=dfxy.y_pred_test, name='Regression Fit'))
    return fig

@callback(Output('output-state', 'children'),
              Input('submit-button', 'n_clicks'),
              Input(component_id=con_dropdown, component_property='value'),
              Input(component_id=mon_dropdown, component_property='value'),
              State('input-1-state', 'value'))
def update_output(n_clicks, selected_con, selected_mon, year):
    lr1, x_test, y_test = lregg(df, selected_con, selected_mon)
    pred_temp_change = lr1.predict([[int(year)]])
    return f'Temperature will increase by {round(pred_temp_change[0][0],3)} degree Celsius'


@callback(Output(component_id='world-graph', component_property='figure'),
              [Input(component_id=mon_dropdown, component_property='value')])
def world_graph(selected_mon):
    df_map = df.copy()
    df_map = df_map[df_map['Months'] == selected_mon]
    df_map['°C'] = ['<=-1.5' if x<=(-1.5) else '<=-1.0' if (-1.5)<x<=(-1.0) else '<=0.0' if (-1.0)<x<=0.0  else '<=0.5' if 0.0<x<=0.5 else '<=1.5' if 0.5<x<=1.5 else '>1.5' if 1.5<=x<10 else 'None' for x in df_map['tem_change']]
    fig = px.choropleth(df_map, locations="Country Code", color="°C", locationmode='ISO-3',hover_name="Country Name",
            hover_data=['tem_change'],animation_frame =df_map.year, labels={'tem_change':'The Temperature Change', '°C':'°C'},
            category_orders={'°C':['<=-1.5','<=-1.0','<=0.0','<=0.5','<=1.5','>1.5','None']},
            color_discrete_map={'<=-1.5':"#08519c",'<=-1.0':"#9ecae1",'<=0.0':"#eff3ff",'<=0.5':"#ffffb2",'<=1.5': "#fd8d3c",'>1.5':"#bd0026",'None':"#252525"},
            title = 'Temperature Change - 1961 - 2019')
    fig.update_layout(autosize=True,template='seaborn', 
                  paper_bgcolor="rgb(234, 234, 242)",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig


@callback(Output(component_id='kworld-graph', component_property='figure'),
              [Input(component_id=mon_dropdown, component_property='value')])
def world_graph(selected_mon):
    df_map = code() 
    df_map = df_map[df_map['Months'] == selected_mon]
    df_map['y_predict'] = ['2' if x==2.0 else '3' if x==0.0 else '1' if x==1.0 else 'None' for x in df_map['y_predict']]

    fig = px.choropleth(df_map, locations="Country Code", color="y_predict", locationmode='ISO-3',hover_name="Country Name",
                hover_data=['tem_change'],animation_frame =df_map.year, labels={'tem_change':'The Temperature Change', 'y-predict':'Zones'},
                category_orders={'Zones':['None','1','2','3']},
                color_discrete_map={'1':"#08519c",'2':"#9ecae1",'3':"#eff3ff", 'None':'#fff'},
                title = 'Worldwide Temperature Change Zones- 1961 - 2019')
    fig.update_layout(autosize=True,template='seaborn', 
                    paper_bgcolor="rgb(234, 234, 242)",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
    