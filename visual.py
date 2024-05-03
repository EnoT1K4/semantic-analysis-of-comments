import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

'''
regex = r"DenseVector\((.*?)\)"
f = open('output.txt','r')
file = open('visualize.txt','w')
for line in f:
    text = re.findall(regex, line)[0].split(', ').pop(1).split(']').pop(0)
    file.write(text + '\n')

'''

df = pd.read_csv('visualize.txt', header = None)
df.columns = ['negative_proba']
df['negative_proba'].astype(float)



share_neg = (df['negative_proba'] > 0.5).sum() / df.shape[0]

print(share_neg)




fig = make_subplots(1,1,
                   subplot_titles=['Распределение комментариев по оценке негативности']     
                   )
fig.add_trace(go.Violin(
    x = df['negative_proba'],
    meanline_visible = True,
    name = '(N = %i)' % df.shape[0],
    side = 'positive',
    spanmode = 'hard'
))


fig.add_annotation(x=0.8, y=1.5,
            text = "%0.2f — доля негативных комментариев (при p > 0.44)"\
                   % share_neg,
            showarrow=False,
            yshift=10)

fig.update_traces(orientation='h', 
                  width = 1.5,
                  points = False
                 )


fig.update_layout(height = 500,
                  #xaxis_showgrid=False,
                  xaxis_zeroline=False,
                  template = 'plotly_dark',
                  font_color = 'rgba(212, 210, 210, 1)',
                  legend=dict(
                    y=0.9,
                    x=-0.1,
                    yanchor='top',
                    ),
                 )
fig.update_yaxes(visible = False)
              


fig.show()