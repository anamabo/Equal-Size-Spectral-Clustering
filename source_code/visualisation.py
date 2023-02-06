"""
Module to visualise the clusters on a map
"""
import logging
import pandas as pd
import numpy as np
import source_code.utils as utils
import plotly.graph_objects as go

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def visualise_clusters(dataset: pd.DataFrame, longitude_colname: str, latitude_colname: str, label_col: str, zoom=11):
    """
    Args:
        dataset: (datasets frame). Dataframe with clustered events. It should contain columns with latitudes, longitudes
                               and the cluster labels.
        longitude_colname (str): the name of the column in dataset containing the longitude of the points.
        latitude_colname (str): the name of the column in dataset containing the latitude of the points.
        label_col (str): the name of the column in dataset containing the cluster label of the points.
        zoom: (int). the zoom to visualize the plot

    Returns:
        A plotly figure instance.
    """
    nclusters = dataset[label_col].nunique()
    colors = utils.colors

    # Create plot with buttons
    buttons = [dict(label="all clusters", method="update", args=[{"visible": np.repeat(True, nclusters)}])]
    fig = go.Figure()
    for cluster in range(nclusters):
        # visible_plot must be 2*nclusters, since you need to show 2 plots per cluster
        visible_plot = np.repeat(False, 2*nclusters)
        j = 2*cluster
        visible_plot[j] = True
        visible_plot[j+1] = True
        buttons.append(dict(label="cluster%d" % cluster, method="update", args=[{'visible': visible_plot}]))

        df = dataset[dataset[label_col] == cluster].copy()
        df.reset_index(drop=True, inplace=True)

        fig.add_trace(go.Scattermapbox(mode="markers",
                                       lat=df[latitude_colname].values,
                                       lon=df[longitude_colname].values,
                                       marker=dict(size=6, color=colors[cluster]),
                                       name='',
                                       )
                      )

        fig.add_trace(go.Scattermapbox(mode="text",
                                       lat=df[latitude_colname].values,
                                       lon=df[longitude_colname].values,
                                       textposition="top center",
                                       name="cluster %d" % cluster,
                                       textfont=dict(size=16),
                                       marker=dict(size=6, color=colors[cluster])
                                       )
                      )

    # Layout of buttons
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
                                        x=1.0, xanchor="right", y=1.1, yanchor="top")])

    fig.update_layout(showlegend=False, title_text="Resulting clusters",
                      mapbox=dict(zoom=zoom,
                                  center={"lat": dataset[latitude_colname].mean(),
                                          "lon": dataset[longitude_colname].mean()},
                                  style="carto-positron"
                                  ),
                      )
    return fig
