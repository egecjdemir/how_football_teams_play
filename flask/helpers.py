import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


def generate_recommender_plots(outcome_matrix_path, outcome_percentages_path):
    t0 = time.time()

    # Plot Outcome Matrix
    sns.set(style="whitegrid")
    pivot_data = folders_dict["outcome_percentages"][outcome_matrix_path].pivot_table(
        index=['BKMeans_Labels', 'Opponent_Cluster'], values=['Draw', 'Lose', 'Win'])

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data.plot(kind='bar', stacked=False, ax=ax)
    ax.set_title('Outcome Probabilities for Each Cluster Against Each Other')
    ax.set_xlabel('(Cluster, Opponent_Cluster)')
    ax.set_ylabel('Probability (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig("static/images/outcome_matrix.png")
    plt.close()

    # Plot Outcome Percentages with Annotations
    fig, ax = plt.subplots(figsize=(10, 6))
    folders_dict["outcome_percentages"][outcome_percentages_path].set_index('BKMeans_Labels')[
        ['Draw', 'Lose', 'Win']].plot(kind='bar', stacked=True, ax=ax)

    for n, x in enumerate([p.get_x() + p.get_width() / 2 for p in ax.patches]):
        height = sum([p.get_height() for p in ax.patches[n::3]])
        for i, p in enumerate(ax.patches[n::3]):
            y = p.get_y() + p.get_height() / 2
            value = int(p.get_height())
            if value != 0:
                ax.text(x, y, f'{value}%', ha='center', va='center')

    ax.set_title('Outcome Probabilities')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Probability (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig("static/images/outcome_percentages.png")
    plt.close()

    # Reshape the data for heatmap
    pivot_draw = folders_dict["outcome_percentages"][outcome_matrix_path].pivot("BKMeans_Labels", "Opponent_Cluster",
                                                                                "Draw")
    pivot_win = folders_dict["outcome_percentages"][outcome_matrix_path].pivot("BKMeans_Labels", "Opponent_Cluster",
                                                                               "Win")

    # Draw heatmaps with correct function call for layout adjustment
    fig, axes = plt.subplots(2, 1, figsize=(12, 18), sharex=True, sharey=True)

    sns.heatmap(pivot_draw, cmap="coolwarm", annot=True, fmt=".1f", ax=axes[0])
    axes[0].set_title('Probability of Draw', fontsize=14)
    axes[0].set_ylabel("Cluster")

    sns.heatmap(pivot_win, cmap="coolwarm", annot=True, fmt=".1f", ax=axes[1])
    axes[1].set_title('Probability of Win', fontsize=14)
    axes[1].set_ylabel("Cluster")

    # Adjust layout
    plt.tight_layout()
    plt.savefig("static/images/outcome_heatmap.png")
    plt.close()

    t1 = time.time()
    print(f"recommend time: {(t1 - t0)}")


def plot_generation_helper(folders_dict, filename, chosen_team_name=None, chosen_league_name = "league"):
    stats = folders_dict["stats_per_cluster"][filename].copy()
    stats.rename(columns={'Unnamed: 0': "teamId"}, inplace=True)

    stats_mean = stats[[col for col in stats.columns if 'mean' in col or col == 'teamId']]

    stats_mean.columns = stats_mean.columns.str.replace('_mean', '')

    cluster_means = stats_mean.iloc[:3]
    labels = list(cluster_means.teamId)
    if len(str(chosen_team_name)) < 1:
        labels += [chosen_team_name]
    else:
        labels += ["Team or League average"]

    print(f"labels: {labels}")

    d = {'Premier League': 'England', 'Bundesliga': 'Germany', 'Serie A': 'Italy', 'Ligue 1': 'France',
         'La Liga': 'Spain'}
    chosen_country = None
    if chosen_league_name in d:
        chosen_country = d[chosen_league_name]

    plots_df = stats_mean.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')

    r = plots_df[plots_df["name"] == chosen_team_name].iloc[:, 1:len(cluster_means.columns)]

    if chosen_country is not None:
        country_df = plots_df.merge(folders_dict["teams_df"]["country_of_teams.csv"])
        filtered_names = country_df[country_df["country"] == chosen_country]['name'].tolist()
        r = plots_df[plots_df.name.isin(filtered_names)].iloc[:, 1:len(cluster_means.columns)]
        r = pd.DataFrame(r.mean()).T

    cluster_means.drop(columns=['teamId'], inplace=True)

    if chosen_team_name is not None or chosen_league_name is not None:
        print(list(r.iloc[0]))
        cluster_means.loc[len(cluster_means)] = list(r.iloc[0])

    return cluster_means, labels


# Function to find the 2 teams above and 2 teams below Liverpool
def find_nearest_teams(team_list, target_team, num_up=2, num_down=2):
    try:
        target_index = team_list.index(target_team)
    except ValueError:
        print(f"{target_team} not found in the team list.")
        return []

    start_index = max(0, target_index - num_up)
    end_index = min(len(team_list), target_index + num_down + 1)

    nearest_teams = team_list[start_index:target_index] + team_list[target_index + 1:end_index]

    # Handle edge cases
    if target_index == 0:
        nearest_teams = team_list[1:1 + num_up + num_down]
    elif target_index == len(team_list) - 1:
        nearest_teams = team_list[-(num_up + num_down + 1):-1]
    elif target_index < num_up:
        nearest_teams = team_list[:target_index] + team_list[target_index + 1:target_index + 1 + num_down]
    elif target_index + num_down >= len(team_list):
        nearest_teams = team_list[target_index - num_up:target_index] + team_list[target_index + 1:]

    return nearest_teams

def stack_plot_df_create(folders_dict, cluster_prob_file = "in_poss_cluster_probs.csv", league = ''):
    #df = folders_dict['cluster_probs'][filename]
    #df = df.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')
    #names = np.array(df["name"])
    #df = df.iloc[:,2:5]

    in_poss_df = folders_dict["cluster_probs"]["in_poss_cluster_probs.csv"]
    out_of_poss_df = folders_dict["cluster_probs"]["out_of_poss_cluster_probs.csv"]
    trans_out_of_poss_df = folders_dict["cluster_probs"]["trans_out_of_poss_cluster_probs.csv"]
    trans_poss_df = folders_dict["cluster_probs"]["trans_poss_cluster_probs.csv"]

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

    team_df = None
    team_labels = None
    if chosen_country:
        df = cluster_probs_df.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')
        df['area_x'] = df['area_x'].apply(ast.literal_eval)
        print("ok")
        team_df = df[df['area_x'].apply(lambda x: x['name'] == chosen_country)]
        print("notok")
        team_labels = team_df["name_x"].tolist()
        team_df = team_df[["0", "1", "2"]]
        print(team_labels)
        print(team_df)

    print(teams_list)




    cluster_probs_df = cluster_probs_df[["0", "1", "2"]]

    print(cluster_probs_df)
    return cluster_probs_df, teams_list, team_df, team_labels

def stack_plotter(df, labels, title, img_path):
    # Sort the DataFrame by the largest stack in each row
    df['Max_Stack'] = df.max(axis=1)
    df_sorted = df.sort_values(by='Max_Stack', ascending=True).drop(columns='Max_Stack')

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a horizontal stacked bar plot
    for i, (index, row) in enumerate(df_sorted.iterrows()):
        ax.barh(labels[i], row[0], color=colors[0], label='Cluster 0' if i == 0 else "")
        ax.barh(labels[i], row[1], left=row[0], color=colors[1], label='Cluster 1' if i == 0 else "")
        ax.barh(labels[i], row[2], left=row[0] + row[1], color=colors[2], label='Cluster 2' if i == 0 else "")

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Values')
    ax.set_ylabel('Labels')

    # Add legend
    ax.legend()

    plt.savefig(img_path)
    plt.close()
