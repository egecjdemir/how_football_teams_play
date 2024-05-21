from flask import Flask, render_template, request, redirect, url_for, jsonify
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from helpers import plot_generation_helper, find_nearest_teams

main_directory = 'data4flask'

# Create an empty dictionary to store dictionaries of DataFrames
folders_dict = {}

# Loop through all folders in the main directory
for folder in os.listdir(main_directory):
    # Construct the full path to the folder
    folder_path = os.path.join(main_directory, folder)

    # Check if the item in the main directory is a folder
    if os.path.isdir(folder_path):
        print("Processing folder:", folder)

        # Create a dictionary to store DataFrames for the current folder
        folder_dataframes = {}

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                # Construct the full file path
                filepath = os.path.join(folder_path, filename)
                print("Reading", filepath)

                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)

                # Add the DataFrame to the folder's dictionary with filename as key
                folder_dataframes[filename] = df

        # Add the folder's dictionary to the main dictionary with folder name as key
        folders_dict[folder] = folder_dataframes

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    if request.method == 'POST':
        if 'start' in request.form:
            team_names = list(folders_dict["teams_df"]["teams_df.csv"].name)
            return render_template('select.html', leagues=['Premier League', 'Bundesliga', 'Serie A', 'Ligue 1', 'La Liga'], team_names=team_names)
    return redirect(url_for('welcome'))


@app.route('/options', methods=['POST'])
def options():
    team = request.form.get('team')
    league = request.form.get('league')
    return render_template('options.html', team=team, league=league)

@app.route('/recommend_playing_style', methods=['POST'])
def recommend_playing_style():
    team = request.form.get('team')
    league = request.form.get('league')
    # Add logic to recommend playing style
    return render_template('recommend_playing_style.html', team=team, league=league)

@app.route('/compare_cluster_statistics', methods=['GET', 'POST'])
def compare_cluster_statistics():
    if request.method == 'POST':
        team = request.form.get('team')
        league = request.form.get('league')

        in_poss_stats = folders_dict["stats_per_cluster"]["in_poss_stats_per_cluster.csv"]
        out_of_poss_stats = folders_dict["stats_per_cluster"]["out_of_poss_stats_per_cluster.csv"]

        in_poss_mean = in_poss_stats[[col for col in in_poss_stats.columns if 'mean' in col or col == 'teamId']]
        in_poss_mean.columns = in_poss_mean.columns.str.replace('_mean', '')

        out_of_poss_mean = out_of_poss_stats[[col for col in out_of_poss_stats.columns if 'mean' in col or col == 'teamId']]
        out_of_poss_mean.columns = out_of_poss_mean.columns.str.replace('_mean', '')

        return render_template('compare_cluster_statistics.html', team=team, league=league, in_poss_features=in_poss_mean.columns, out_of_poss_features=out_of_poss_mean.columns)
    if request.method == 'GET':
        return redirect(url_for('welcome'))


@app.route('/visualize_clustering', methods=['POST'])
def visualize_clustering():
    team = request.form.get('team')
    league = request.form.get('league')
    if not team and not league:
        return redirect(url_for('select'))
    return render_template('visualize_clustering.html', team=team, league=league)


@app.route('/check_cluster_probabilities', methods=['POST'])
def check_cluster_probabilities():
    team = request.form.get('team')
    league = request.form.get('league')
    # Add logic to check cluster probabilities
    return render_template('check_cluster_probabilities.html', team=team, league=league)


@app.route('/plots', methods=['GET', 'POST'])
def plots():
    if request.method == 'POST':
        team = request.form.get('team')
        league = request.form.get('league')
        if team and league:
            return redirect(url_for('select'))
        elif team or league:
            return render_template('visualize_clustering.html', team=team, league=league)
    return redirect(url_for('select'))

@app.route('/dendrogram', methods=['POST'])
def dendrogram_plot():
    try:
        data = request.get_json()
        chosen_league_name = data.get('league', '')
        chosen_team_name = data.get('team', '')

        d = {'Premier League': 'England', 'Bundesliga': 'Germany', 'Serie A': 'Italy', 'Ligue 1': 'France', 'La Liga': 'Spain'}
        chosen_country = None
        if chosen_league_name in d:
            chosen_country = d[chosen_league_name]

        dendo_df = folders_dict["pca_scatter"]["final_df.csv"].merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')
        filtered_names = []
        if chosen_country is not None:
            country_df = dendo_df.merge(folders_dict["teams_df"]["country_of_teams.csv"])
            filtered_names = country_df[country_df["country"] == chosen_country]['name'].tolist()

        print(f"filtered_names: {filtered_names}")

        dendo_names = dendo_df['name'].tolist()
        dendo_df = dendo_df.iloc[:, :15].drop(["teamId", "labels", "names"], axis=1)


        print(f"chosen_league_name: {chosen_league_name}")
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
            team_names = [label.get_text() for label in x_labels]
            for label in x_labels:
                if label.get_text() == chosen_team_name:
                    label.set_color('red')
                    label.set_fontweight('bold')
            nearest_teams = find_nearest_teams(team_names, chosen_team_name, 2, 2)
            plt.suptitle(f'The nearest teams to {chosen_team_name} are: {nearest_teams}', fontsize=12)

        elif filtered_names is not None:
            ax = plt.gca()
            x_labels = ax.get_xmajorticklabels()
            for label in x_labels:
                if label.get_text() in filtered_names:
                    label.set_color('red')
                    label.set_fontweight('bold')


        # Beautify the dendrogram
        plt.title(f'Dendrogram of Teams', fontsize=20)
        plt.xlabel('Teams', fontsize=15)
        plt.ylabel('Euclidean distance (Ward)', fontsize=15)
        plt.grid(True)

        plt.savefig("static/images/dendro.png", bbox_inches='tight')
        plt.close()

        # Generate the scatter plot
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dendo_df)

        k = 3
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans_labels = kmeans.fit_predict(scaled_data)

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        plt.figure(figsize=(15, 10))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, s=50, cmap='jet')

        for i, team_name in enumerate(dendo_names):
            plt.annotate(team_name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=7, alpha=0.75)
            if team_name == chosen_team_name or team_name in filtered_names:
                #plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color='orange', s=200, edgecolor='black',
                #            label=chosen_team_name)
                plt.annotate(team_name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=7, color='red', alpha=0.9)

        plt.title('K-means Clustering with PCA-Reduced Data', fontsize=15)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)

        plt.savefig("static/images/scatter.png", bbox_inches='tight')
        plt.close()

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error generating dendrogram and scatter plot: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate_bar_plots', methods=['POST'])
def generate_bar_plots():
    try:
        data = request.get_json()
        chosen_team_name = data.get('team', '')
        feature = data.get('feature', '')
        poss_type = data.get('poss_type', '')

        if 'in_poss' in poss_type:
            in_poss_stats, in_poss_labels = plot_generation_helper(folders_dict, 'in_poss_stats_per_cluster.csv', chosen_team_name)
            trans_in_poss_stats, trans_in_poss_labels = plot_generation_helper(folders_dict, 'trans_in_poss_stats_per_cluster.csv', chosen_team_name)

            plt.figure(figsize=(12, 6))
            ax = plt.subplot(121)
            sns.barplot(x=in_poss_labels, y=in_poss_stats[feature], ax=ax)
            ax.set_title(f'{feature}')
            ax.set_xticklabels(in_poss_labels, rotation=45, ha='right')

            ax = plt.subplot(122)
            sns.barplot(x=trans_in_poss_labels, y=trans_in_poss_stats[feature], ax=ax)
            ax.set_title(f'{feature} (In Transition)')
            ax.set_xticklabels(trans_in_poss_labels, rotation=45, ha='right')

            plt.savefig(f"static/images/{feature}_in_poss_bar_plot.png", bbox_inches='tight')
            plt.close()

        if 'out_of_poss' in poss_type:
            out_of_poss_stats, out_of_poss_labels = plot_generation_helper(folders_dict, 'out_of_poss_stats_per_cluster.csv', chosen_team_name)
            trans_out_of_poss_stats, trans_out_of_poss_labels = plot_generation_helper(folders_dict, 'trans_out_of_poss_stats_per_cluster.csv', chosen_team_name)

            plt.figure(figsize=(15, 10))
            ax = plt.subplot(121)
            sns.barplot(x=out_of_poss_labels, y=out_of_poss_stats[feature], ax=ax)
            ax.set_title(f'{feature}')
            ax.set_xticklabels(out_of_poss_labels, rotation=45, ha='right')

            ax = plt.subplot(122)
            sns.barplot(x=trans_out_of_poss_labels, y=trans_out_of_poss_stats[feature], ax=ax)
            ax.set_title(f'{feature} (In Transition)')
            ax.set_xticklabels(trans_out_of_poss_labels, rotation=45, ha='right')

            plt.savefig(f"static/images/{feature}_out_of_poss_bar_plot.png", bbox_inches='tight')
            plt.close()

        return jsonify({"status": "success", "feature": feature, "poss_type": poss_type}), 200
    except Exception as e:
        print(f"Error generating bar plots: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)