import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    
    df_cat = df[['active', 'alco', 'cardio', 'cholesterol', 'gluc', 'overweight', 'smoke']]


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat_long = pd.melt(df_cat, id_vars=['cardio'], value_name='total')
    df_cat_long['variable'] = pd.Categorical(df_cat_long['variable'])

    # Draw the catplot with 'sns.catplot()'



    # Get the figure for the output
    fig = sns.catplot(x='variable', kind='count', hue='total', col='cardio', data=df_cat_long)
    fig.set_ylabels("total")
    fig.set_xlabels("variable")

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df = pd.read_csv("medical_examination.csv")
    df = df.loc[df['ap_lo'] <= df['ap_hi']]
    df = df.loc[df['height'] >= df['height'].quantile(0.025)]
    df = df.loc[df['height'] <= df['height'].quantile(0.975)]
    df = df.loc[df['weight'] >= df['weight'].quantile(0.025)]
    df = df.loc[df['weight'] <= df['weight'].quantile(0.975)]
    df_heat = df

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)
    




    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(data=corr, mask=mask, annot=True, fmt=".1f", ax=ax)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
