import seaborn as sbn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",20)

df = pd.read_csv("F:/ML_Projects/AmazonRecommendation/train.csv")

print(df.info())

# first remove outliers from affinity

q75, q25 = np.percentile(df['customer_affinity_score'],[75,25])
left = q25-(1.5*(q75-q25))
right = q75+(1.5*(q75-q25))
df_NoOUT = df.loc[(df['customer_affinity_score']<=right) & (df['customer_affinity_score']>=left)]


#----------Orders vs affinity------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.lmplot(x='customer_affinity_score', y='customer_order_score',hue='customer_category',
           data=df_NoOUT,markers=['o','x'],fit_reg=False, legend_out=True)

plt.ylabel('Order Score')
plt.xlabel('Affinity')
plt.show()


#--------for this we can not have any none values in leaf-----
df_NoNullActive = df.dropna(subset=['customer_active_segment','X1'])
fig = px.sunburst(df_NoNullActive,path=['customer_category','customer_active_segment','X1'],
                 color_discrete_sequence=px.colors.qualitative.Safe)
fig.show()

#-------visit score------------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.histplot(data=df,x='customer_visit_score',hue='customer_category',kde=True)
plt.xlabel('Visit Score')
plt.ylabel('Count')
plt.show()

# ----------product search vs clicks on searched links----------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.lmplot(x='customer_product_search_score', y='customer_ctr_score',hue='customer_category',
           data=df,markers=['o','x'],fit_reg=False, legend_out=True)

plt.ylabel('Searched Links Click Score')
plt.xlabel('Product Search Score')
plt.show()

# ----------variety search vs clicks on searched links----------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.lmplot(x='customer_product_variation_score', y='customer_ctr_score',hue='customer_category',
           data=df,markers=['o','x'],fit_reg=False, legend_out=True)

plt.ylabel('Searched Links Click Score')
plt.xlabel('Variety Search Score')
plt.show()

# # ----------stay score vs visiting freqency----------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.lmplot(x='customer_visit_score', y='customer_stay_score',hue='customer_category',
           data=df,markers=['o','x'],fit_reg=False, legend_out=True)

plt.ylabel('Stay Score')
plt.xlabel('Frequency Score')
plt.show()


# -------order score------------
colors = ["#A4C3D2","#8B0000"]
sbn.set_palette(sbn.color_palette(colors))
sbn.set_style('whitegrid')
sbn.histplot(data=df,x='customer_order_score',hue='customer_category',kde=True)
plt.xlabel('Order Score')
plt.ylabel('Count')
plt.show()

