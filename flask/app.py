from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import time

# Specify the directory path containing folders with CSV files
main_directory = 'data4flask'

# Create an empty dictionary to store DataFrames
dataframes_dict = {}

# Loop through all folders in the main directory
for folder in os.listdir(main_directory):
    # Construct the full path to the folder
    folder_path = os.path.join(main_directory, folder)

    # Check if the item in the main directory is a folder
    if os.path.isdir(folder_path):

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                # Construct the full file path
                filepath = os.path.join(folder_path, filename)

                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)

                # Add the DataFrame to the dictionary with filename as key
                dataframes_dict[filename] = df

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    if request.method == 'POST':
        if 'start' in request.form:
            team_names = list(dataframes_dict["teams_df.csv"].name)
            return render_template('select.html', leagues=['PL', 'Bundesliga', 'Serie A', 'Ligue 1', 'La Liga'], team_names=team_names)
    return redirect(url_for('welcome'))


@app.route('/plots', methods=['GET', 'POST'])
def plots():
    if request.method == 'POST':
        team = request.form.get('team')
        league = request.form.get('league')
        if team and league:
            return redirect(url_for('select'))
        elif team or league:
            return render_template('plots.html', team=team, league=league)
    return redirect(url_for('select'))

@app.route('/dendrogram', methods=['POST'])
def dendrogram_plot():
    dendo_df = dataframes_dict["final_df.csv"].merge(dataframes_dict["teams_df.csv"], left_on='teamId', right_on='wyId')
    dendo_names = dendo_df['name'].tolist()
    dendo_df = dendo_df.iloc[:, :15].drop(["teamId", "labels", "names"], axis=1)

    data = request.get_json()
    chosen_team_name = data.get('team', '')
    print(f"Chosen team name: {chosen_team_name}")

    # Generate the linkage matrix
    linkage_matrix = linkage(dendo_df, method='ward')

    # Create the dendrogram
    plt.figure(figsize=(15, 10))
    dendro = dendrogram(
        linkage_matrix,
        labels=dendo_names,
        leaf_rotation=90,
        leaf_font_size=12
    )

    # Highlight chosen team name in red if any
    if chosen_team_name in dendo_names:
        ax = plt.gca()
        x_labels = ax.get_xmajorticklabels()
        for label in x_labels:
            if label.get_text() == chosen_team_name:
                label.set_color('red')
                label.set_fontweight('bold')

    # Beautify the dendrogram
    plt.title('Dendrogram of Teams', fontsize=20)
    plt.xlabel('Teams', fontsize=15)
    plt.ylabel('Euclidean distance (Ward)', fontsize=15)
    plt.grid(True)

    plt.savefig("static/images/dendro.png", bbox_inches='tight')
    plt.close()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)