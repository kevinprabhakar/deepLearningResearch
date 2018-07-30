import pandas as pd
import numpy as np
from scipy import stats
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import codecs
import json
import csv

plotly.tools.set_credentials_file(username='Jitorew', api_key='0X8WMlENLwVrYzdVPdDH')


file_path = "frequencyTables/pcaResultOnFreqTableIndividualWords.json"
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)

variance = stats.describe(a_new).variance
min = stats.describe(a_new).minmax[0]
max = stats.describe(a_new).minmax[1]
average = stats.describe(a_new).mean

candlestick = go.Candlestick(x = range(0,len(a_new)),
                       low = min,
                       high = max,
                       open = average,
                       close = average)

varianceLine = go.Scatter(x = range(0, len(a_new)), y = variance,line=dict(color='rgb(51,153,255)'), name="Variance")


data = [candlestick, varianceLine]
py.iplot(data, filename='min_max_mean')