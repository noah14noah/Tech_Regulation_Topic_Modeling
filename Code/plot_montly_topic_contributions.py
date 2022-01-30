#Data visualization for average summed topic contributons
# packages
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
import numpy as np
import sys

# parameters
num_topics = 30
s_year = 1900
e_year = 2020 + 1


# import
monthly_contributions = pd.read_csv(
    "/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/monthly_topic_contribution")

scatterplot = px.scatter(data_frame=monthly_contributions, x=len(monthly_contributions.columns), y=monthly_contributions.iloc[[0]])

scatterplot.show()