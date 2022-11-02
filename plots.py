from __future__ import annotations

from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_files = glob("results/*/*/*/results.json")


dfs: list[pd.DataFrame] = []
for file in results_files:
    optimized_player, method = file.split("/")[1:3]
    model_type, battle_agent = file.split("/")[3].split("-")[1:3]

    raw_df = pd.DataFrame(pd.read_json(file))
    player_index = 0 if optimized_player == "first" else 1
    raw_series = raw_df.iloc[player_index]

    df = pd.DataFrame(
        {"episode": raw_series.checkpoints, "win_rate": raw_series.win_rates}
    )
    df = df.assign(
        **raw_series.drop(
            columns=[
                "action_histograms",
                "checkpoints",
                "ep_lengths",
                "tensorboard_log",
                "win_rates",
            ],
        )
    )
    df["player"] = optimized_player
    df["method"] = method
    df["model_type"] = model_type
    df["battle_agent"] = battle_agent
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.win_rate *= 100
df.episode /= 1000
df.loc[df["model_type"] == "lstm", "model_type"] = "LSTM"
df.loc[df["model_type"] == "immediate", "model_type"] = "Immediate"
df.loc[df["battle_agent"] == "max", "battle_agent"] = "max-attack"

sns.set_theme()

fig_players = sns.relplot(
    x="episode",
    y="win_rate",
    kind="line",
    hue="method",
    col="player",
    data=df,
)
fig_players.set_titles("Win rate of the {col_name} player network")

fig_average = sns.relplot(
    x="episode",
    y="win_rate",
    kind="line",
    hue="method",
    data=df,
)
fig_average.ax.set_title("Average")

for fig in [fig_players, fig_average]:
    fig.set_axis_labels("Episodes", "Win rate (%)")
    fig.legend.set_title("")
    for ax in fig.fig.axes:
        ax.set_ylim(35, 105)
        ax.xaxis.set_major_formatter("{x:.0f}k")
    fig.tight_layout()

plt.show()
