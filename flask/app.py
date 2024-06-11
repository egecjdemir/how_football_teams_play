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

import ast

from helpers import plot_generation_helper, find_nearest_teams, stack_plot_df_create, stack_plotter

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
    try:
        team = request.form.get('team')
        league = request.form.get('league')

        # Load the teams DataFrame
        teams_df = folders_dict["teams_df"]["teams_df.csv"]

        # Get the team ID
        team_id = None
        if team:
            team_id = teams_df.loc[teams_df['name'] == team, 'wyId'].values[0]

        # Helper function to get the highest probability cluster
        def get_highest_prob_cluster(probs_df, team_id):
            team_probs = probs_df[probs_df['teamId'] == team_id]
            if not team_probs.empty:
                team_probs = team_probs.drop(columns=['teamId', 'Unnamed: 0'])
                max_prob_cluster = team_probs.idxmax(axis=1).values[0]
                return max_prob_cluster
            return None

        # Load cluster probabilities DataFrames
        in_poss_probs = folders_dict["cluster_probs"]["in_poss_cluster_probs.csv"]
        out_of_poss_probs = folders_dict["cluster_probs"]["out_of_poss_cluster_probs.csv"]
        trans_in_poss_probs = folders_dict["cluster_probs"]["trans_poss_cluster_probs.csv"]
        trans_out_of_poss_probs = folders_dict["cluster_probs"]["trans_out_of_poss_cluster_probs.csv"]

        # Get highest probability clusters for each phase
        in_poss_cluster = get_highest_prob_cluster(in_poss_probs, team_id)
        out_of_poss_cluster = get_highest_prob_cluster(out_of_poss_probs, team_id)
        trans_in_poss_cluster = get_highest_prob_cluster(trans_in_poss_probs, team_id)
        trans_out_of_poss_cluster = get_highest_prob_cluster(trans_out_of_poss_probs, team_id)

        # Generate the plots
        t0 = time.time()

        datasets = ['final', 'in_poss', 'out_of_poss', 'trans_in_poss', 'trans_out_of_poss']
        for dataset in datasets:
            outcome_matrix_path = f"{dataset}_outcome_matrix.csv"
            outcome_percentages_path = f"{dataset}_outcome_percentages.csv"

            if outcome_matrix_path in folders_dict["outcome_percentages"] and outcome_percentages_path in folders_dict[
                "outcome_percentages"]:
                label = 'BKMeans_Labels' if dataset == 'final' else 'KMeans_Labels'

                # Plot Outcome Matrix
                sns.set(style="whitegrid")
                pivot_data = folders_dict["outcome_percentages"][outcome_matrix_path].pivot_table(
                    index=[label, 'Opponent_Cluster'], values=['Draw', 'Lose', 'Win'])

                fig, ax = plt.subplots(figsize=(12, 8))
                pivot_data.plot(kind='bar', stacked=False, ax=ax)
                ax.set_title(f'Outcome Probabilities for Each Cluster Against Each Other ({dataset})')
                ax.set_xlabel(f'({label}, Opponent_Cluster)')
                ax.set_ylabel('Probability (%)')
                plt.xticks(rotation=45)
                plt.legend(title='Outcome')
                plt.tight_layout()
                plt.savefig(f"static/images/{dataset}_outcome_matrix.png")
                plt.close()

                # Plot Outcome Percentages with Annotations
                fig, ax = plt.subplots(figsize=(10, 6))
                folders_dict["outcome_percentages"][outcome_percentages_path].set_index(label)[
                    ['Draw', 'Lose', 'Win']].plot(kind='bar', stacked=True, ax=ax)

                for n, x in enumerate([p.get_x() + p.get_width() / 2 for p in ax.patches]):
                    height = sum([p.get_height() for p in ax.patches[n::3]])
                    for i, p in enumerate(ax.patches[n::3]):
                        y = p.get_y() + p.get_height() / 2
                        value = int(p.get_height())
                        if value != 0:
                            ax.text(x, y, f'{value}%', ha='center', va='center')

                ax.set_title(f'Outcome Probabilities ({dataset})')
                ax.set_xlabel('Clusters')
                ax.set_ylabel('Probability (%)')
                plt.xticks(rotation=0)
                plt.legend(title='Outcome')
                plt.tight_layout()
                plt.savefig(f"static/images/{dataset}_outcome_percentages.png")
                plt.close()

                # Reshape the data for heatmap
                pivot_draw = folders_dict["outcome_percentages"][outcome_matrix_path].pivot(label, "Opponent_Cluster",
                                                                                            "Draw")
                pivot_win = folders_dict["outcome_percentages"][outcome_matrix_path].pivot(label, "Opponent_Cluster",
                                                                                           "Win")

                # Draw heatmaps with correct function call for layout adjustment
                fig, axes = plt.subplots(2, 1, figsize=(12, 18), sharex=True, sharey=True)

                sns.heatmap(pivot_draw, cmap="coolwarm", annot=True, fmt=".1f", ax=axes[0])
                axes[0].set_title(f'Probability of Draw ({dataset})', fontsize=14)
                axes[0].set_ylabel("Cluster")

                sns.heatmap(pivot_win, cmap="coolwarm", annot=True, fmt=".1f", ax=axes[1])
                axes[1].set_title(f'Probability of Win ({dataset})', fontsize=14)
                axes[1].set_ylabel("Cluster")

                plt.tight_layout()
                plt.savefig(f"static/images/{dataset}_outcome_heatmap.png")
                plt.close()

        t1 = time.time()
        print(f"recommend time: {(t1 - t0)}")

        return render_template('recommend_playing_style.html', team=team, league=league,
                               in_poss_cluster=in_poss_cluster, out_of_poss_cluster=out_of_poss_cluster,
                               trans_in_poss_cluster=trans_in_poss_cluster,
                               trans_out_of_poss_cluster=trans_out_of_poss_cluster,
                               datasets=datasets)
    except Exception as e:
        print(f"Error generating playing style recommendations: {e}")
        return render_template('error.html', message="An error occurred while processing your request.")


@app.route('/compare_cluster_statistics', methods=['GET', 'POST'])
def compare_cluster_statistics():
    t0 = time.time()
    if request.method == 'POST':
        team = request.form.get('team')
        league = request.form.get('league')

        if len(str(team)) < 1:
            print("wpeıfjpwıefjwpıwpfejwıejfowjeofwjke")
            team = None

        in_poss_stats = folders_dict["stats_per_cluster"]["in_poss_stats_per_cluster.csv"]
        out_of_poss_stats = folders_dict["stats_per_cluster"]["out_of_poss_stats_per_cluster.csv"]

        in_poss_mean = in_poss_stats[[col for col in in_poss_stats.columns if 'mean' in col or col == 'teamId']]
        in_poss_mean.columns = in_poss_mean.columns.str.replace('_mean', '')

        out_of_poss_mean = out_of_poss_stats[[col for col in out_of_poss_stats.columns if 'mean' in col or col == 'teamId']]
        out_of_poss_mean.columns = out_of_poss_mean.columns.str.replace('_mean', '')

        t1 = time.time()
        print(f"cluster_stats time: {(t1 - t0)}")
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


@app.route('/check_cluster_probabilities', methods=['GET', 'POST'])
def check_cluster_probabilities():
    t0 = time.time()
    in_poss_df = folders_dict["cluster_probs"]["in_poss_cluster_probs.csv"]
    out_of_poss_df = folders_dict["cluster_probs"]["out_of_poss_cluster_probs.csv"]
    trans_out_of_poss_df = folders_dict["cluster_probs"]["trans_out_of_poss_cluster_probs.csv"]
    trans_poss_df = folders_dict["cluster_probs"]["trans_poss_cluster_probs.csv"]

    if request.method == 'POST':
        try:
            team = request.form.get('team')
            league = request.form.get('league')
            cluster_prob_file = request.form.get('cluster_prob_file')

            cluster_probs_df = in_poss_df
            if cluster_prob_file == "out_of_poss_cluster_probs.csv":
                cluster_probs_df = out_of_poss_df
            elif cluster_prob_file == "trans_out_of_poss_cluster_probs.csv":
                cluster_probs_df = trans_out_of_poss_df
            elif cluster_prob_file == "trans_poss_cluster_probs.csv":
                cluster_probs_df = trans_poss_df

            cluster_probs_df = cluster_probs_df.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')
            teams_list = cluster_probs_df["name"].tolist()

            d = {'Premier League': 'England', 'Bundesliga': 'Germany', 'Serie A': 'Italy', 'Ligue 1': 'France',
                 'La Liga': 'Spain'}
            chosen_country = None
            if league in d:
                chosen_country = d[league]

            if chosen_country:
                df = cluster_probs_df.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')
                df['area_x'] = df['area_x'].apply(ast.literal_eval)
                cluster_probs_df = df[df['area_x'].apply(lambda x: x['name'] == chosen_country)]
                teams_list = cluster_probs_df["name_x"].tolist()

            t1 = time.time()
            print(f"cluster prob time: {(t1 - t0)}")

            return render_template('check_cluster_probabilities.html', team=team, league=league, teams_list=teams_list, cluster_probs_df=cluster_probs_df)
        except Exception as e:
            return render_template('error.html', message="An error occurred while processing your request.")

    if request.method == 'GET':
        return redirect(url_for('welcome'))





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
        t0 = time.time()
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
        t5 = time.time()
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

        t1 = time.time()
        print(f"dendrogram time: {(t1 - t5)}")
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

        t3 = time.time()
        print(f"scatter time: {(t3 - t1)}")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error generating dendrogram and scatter plot: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate_bar_plots', methods=['POST'])
def generate_bar_plots():
    try:
        data = request.get_json()
        chosen_league_name = data.get('league', '')
        chosen_team_name = data.get('team', '')
        feature = data.get('feature', '')
        poss_type = data.get('poss_type', '')

        if chosen_team_name != None:
            print("q3pıfjqıopwrfj0qğeı")

        print(f"Received request for team: {chosen_team_name}, league: {chosen_league_name}, feature: {feature}, poss_type: {poss_type}")


        if 'in_poss' in poss_type:
            in_poss_stats, in_poss_labels = plot_generation_helper(folders_dict, 'in_poss_stats_per_cluster.csv', chosen_team_name, chosen_league_name)
            trans_in_poss_stats, trans_in_poss_labels = plot_generation_helper(folders_dict, 'trans_in_poss_stats_per_cluster.csv', chosen_team_name, chosen_league_name)

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
            out_of_poss_stats, out_of_poss_labels = plot_generation_helper(folders_dict, 'out_of_poss_stats_per_cluster.csv', chosen_team_name, chosen_league_name)
            trans_out_of_poss_stats, trans_out_of_poss_labels = plot_generation_helper(folders_dict, 'trans_out_of_poss_stats_per_cluster.csv', chosen_team_name, chosen_league_name)

            print(f"out_of_poss_stats columns: {out_of_poss_stats.columns}")
            print(f"trans_out_of_poss_stats columns: {trans_out_of_poss_stats.columns}")

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


@app.route('/generate_cluster_prob_plots', methods=['POST'])
def generate_cluster_prob_plots():
    try:
        data = request.get_json()
        chosen_team_name = data.get('team', 'None')
        chosen_league_name = data.get('league', 'None')
        cluster_prob_file = data.get('cluster_prob_file')

        df, labels, league_df, league_labels = stack_plot_df_create(folders_dict, cluster_prob_file, chosen_league_name)

        stack_plotter(df, labels, "Cluster Probabilities", 'static/images/all_teams_prob_plot.png')
        if league_df:
            stack_plotter(league_df, league_labels , f"Cluster Probabilities - {cluster_prob_file}", 'static/images/league_teams_prob_plot.png')

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error generating cluster probability plots: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=8000)