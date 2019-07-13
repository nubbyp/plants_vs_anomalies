import flask
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from flask import request
import pandas as pd
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from keras.models import load_model
import tensorflow as tf
import awstext

mean = 0.3928570407726534
deviant = 0.3396423645323013
deviation = 1.90

df=pd.DataFrame(columns=[
	'co2_1', 'co2_2', 'co2_3', 'co2_4', 'temp_1', 'temp_2', 'temp_3',
       'temp_4', 'dew_1', 'dew_2', 'dew_3', 'dew_4', 'relH_1', 'relH_2',
       'relH_3', 'relH_4', 'externTemp_1', 'externHumid_1',
       'externCondition_1', 'hourIndex', 'dayIndex', 'anomaly'])

cols = [
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'temp_1', 'temp_2', 'temp_3',
       'temp_4', 'dew_1', 'dew_2', 'dew_3', 'dew_4', 'relH_1', 'relH_2',
       'relH_3', 'relH_4', 'externTemp_1', 'externHumid_1',
       'externCondition_1', 'hourIndex', 'dayIndex'
]

server = flask.Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/'
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(src='../static/Plants_vs.jpg'),
                dcc.Slider(
                    id='deviation-slider',
                    min=1,
                    max=10,
                    step=0.1,
                    value=deviation
                ),
                html.P(
                    deviation,
                    id='deviation-val'
                )
            ]
        ),
        html.Div(
            [
                dcc.Graph(id='live-graph',animate=True),
                dcc.Interval(
                    id='graph-update',
                    interval=1*1000,
                    n_intervals=0
                ),
            ]
        )
    ]
)

def infer(inference_base):
    global model
    global deviation
    global mean
    global deviant
    #print('BASE:',inference_base)
    preds = model.predict(inference_base)
    mse = np.mean(np.power(np.array(inference_base) - preds, 2))
    print('mse', mse)
    #results = {}
    #for col in cols:
    #    results[col] = inference_base[col]
    results = pd.DataFrame(inference_base)
    #print('R1:',results)
    is_anomaly = (mse>(mean+deviation*deviant))
    results['anomaly'] = mse
    #anomalies = pd.Series((mse>(mean+deviation*deviant)).astype(int))
    #results['anomaly'] = 1 in set(anomalies)
    #print('anomalies:', anomalies)
    #print('anomaly col:', results['anomaly'])
    #tf=results['anomaly'].astype(int).sum()>0
    print('*** ANOMALY!!' if is_anomaly else "Not anomaly")
    if is_anomaly:
        on_anomaly()
    print('R2:',results)
    return results

def on_anomaly():
    print("Anomaly")
    #awstext.publish("Suspicious values")

@server.route('/update/',methods=['POST'])
def df_update():
    global df
    df=df.append(infer(pd.read_json(request.json)),ignore_index=True)
    return 'Thanks for the lovely data!'


@app.callback(
    dash.dependencies.Output('deviation-val', 'children'),
    [dash.dependencies.Input('deviation-slider', 'value')])
def update_deviation(n):
    global deviation
    deviation = n
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Img(src='../static/Plants_vs.jpg'),
                    dcc.Slider(
                        id='deviation-slider',
                        min=1,
                        max=10,
                        step=0.1,
                        value=n
                    ),
                    html.P(
                        deviation,
                        id='deviation-val'
                    )
                ], width=500
            ),
            html.Div(
                [
                    dcc.Graph(id='live-graph',animate=True),
                    dcc.Interval(
                        id='graph-update',
                        interval=1*1000,
                        n_intervals=0
                    ),
                ]
            )
        ]
    )
    update_graph_scatter(0)
    return deviation

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'interval')])
def update_graph_scatter(n):
    global df
    global mean
    global deviation
    global deviant
    print(deviation)
    working_df = df.tail(96) # Get past 24 hrs dataframe
    fig = tools.make_subplots(rows=working_df.shape[1], cols=1, print_grid=False,
          #  specs = [[{}]] * df.shape[1],
           # shared_xaxes=True, 
       #     vertical_spacing=0.1,
        )
    for (i, col) in enumerate(df.columns):
        val = None
        if(col == 'anomaly'):
            val = [(mse>(mean+deviation*deviant)) for mse in working_df[col]]
        else:
            val = working_df[col]
        fig.append_trace(go.Scatter(
            x=np.array(working_df.index),
            y=val,
            name=col
        ), i+1, 1)
    fig['layout'].update(height=1000, width=1000)
    return fig


if __name__ == '__main__':
    global model
    tf.keras.backend.clear_session()
    model = load_model('autoencoder_north_plants.h5')
    model._make_predict_function()
    app.run_server()
