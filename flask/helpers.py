import pandas as pd
import numpy as np

def plot_generation_helper(folders_dict, filename, chosen_team_name=None):
    stats = folders_dict["stats_per_cluster"][filename].copy()
    stats.rename(columns={'Unnamed: 0': "teamId"}, inplace=True)

    stats_mean = stats[[col for col in stats.columns if 'mean' in col or col == 'teamId']]

    stats_mean.columns = stats_mean.columns.str.replace('_mean', '')

    cluster_means = stats_mean.iloc[:3]
    labels = list(cluster_means.teamId)
    if chosen_team_name != None:
        labels += [chosen_team_name]
    plots_df = stats_mean.merge(folders_dict["teams_df"]["teams_df.csv"], left_on='teamId', right_on='wyId')

    r = plots_df[plots_df["name"] == chosen_team_name].iloc[:, 1:len(cluster_means.columns)]
    cluster_means.drop(columns=['teamId'], inplace=True)

    if chosen_team_name is not None:
        print("Cluster Means Columns:", cluster_means.columns)
        print("Cluster Means Columns:", len(cluster_means.columns))
        print("Row Columns:", r.columns)
        print("Row Columns:", len(r.columns))
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