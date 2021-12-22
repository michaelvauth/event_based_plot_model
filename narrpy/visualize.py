import importlib
import pandas as pd
import numpy as np
import re
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import narrpy.narrativity_graph_class as ngc


def format_annotation_text(text: str) -> str:
    while '  ' in text:
        text = text.replace('  ', ' ')

    text_list = text.split(' ')
    output_string = "<I><br>"
    for item in range(0, 60, 10):
        output_string += ' '.join(text_list[item: item + 10]) + '<br>'

    if len(text_list) > 60:
        output_string += '[...]'

    output_string += "</I>"
    return output_string


def find_id_name(sum_id: str) -> str:
    if '_' in sum_id:
        return re.findall(r'(.*?)_', sum_id)[0]
    else:
        return sum_id


def transform_ids(ids: list) -> list:
    id_dict = {sum_id: index for index, sum_id in enumerate(set(ids))}
    new_ids_list = [
        # f'Annotator {id_dict[item] + 1}' for item in ids
        id_dict[item] + 1 for item in ids
    ]
    return new_ids_list


def duplicate_rows(ac_df: pd.DataFrame, property_col: str) -> pd.DataFrame:
    """
    Duplicates rows in DataFrame if multiple property values exist in defined porperty column.
    """
    def duplicate_generator(df: pd.DataFrame):
        for _, row in df.iterrows():
            if len(row[property_col]) > 1:
                for item in row[property_col]:
                    row_dict = dict(row)
                    row_dict[property_col] = item
                    yield row_dict
            else:
                row_dict = dict(row)
                if len(row[property_col]) > 0:
                    row_dict[property_col] = row[property_col][0]
                    yield dict(row_dict)
                else:
                    row_dict[property_col] = np.nan
                    yield dict(row_dict)

    df_new = pd.DataFrame(list(duplicate_generator(ac_df)))
    return df_new


def plot_student_summaries(summaries_file: pd.DataFrame) -> go.Figure:
    sum_df = pd.read_json(summaries_file)

    # add sum features for the interactive scatter plot
    sum_df = duplicate_rows(sum_df, property_col='prop:summary_index')

    sum_df['prop:summary_index'] = sum_df['prop:summary_index'].apply(
        find_id_name)
    sum_df['prop:summary_index'] = transform_ids(
        list(sum_df['prop:summary_index']))

    sum_df.sort_values('prop:summary_index', inplace=True)

    sum_df['annotation'] = sum_df['annotation'].apply(format_annotation_text)
    sum_df['size'] = (sum_df['end_point'] - sum_df['start_point'])

    title = re.findall(r'summaries_(.*?)\.', summaries_file)[0].upper()
    fig = px.scatter(
        sum_df,
        x='start_point',
        y='prop:summary_index',
        size='size',
        hover_data=['annotation'],
        title=title,
        width=1400
    )

    return fig


def summary_histogramm(directory: str):
    for file in os.listdir(directory):
        event_data = pd.read_json(directory + file)
        title = re.findall(r'summary_(.*?)\.', file)[0].upper()
        fig = px.histogram(
            event_data,
            x='summary_frequency',
            title=title
        )
        fig.show()


def narrativity_graph_example(summary_data_file: str):
    smoothing_windows = [20, 100]
    narr_values = [
        [0, 1, 1, 1],
        [0, 1, 2, 3],
        [0, 2, 5, 7],
        [0, 0, 50, 50],
    ]
    narr_types = ['non_event', 'stative_event', 'process', 'change_of_state']

    ng = ngc.NarrativityGraph(
        event_annotations=pd.read_json(summary_data_file),
        abs_smoothing_windows=smoothing_windows
    )

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            f'Smoothing Window: {window}<br>Narrativity Scale: {narr_value}'
            for window in smoothing_windows
            for narr_value in narr_values
        ]
    )

    for value_index, values in enumerate(narr_values):
        narr_dict = dict(zip(narr_types, values))
        ng.compute_narrativity_values(
            tag_values=narr_dict, abs_smoothing_windows=smoothing_windows)
        for window_index, window in enumerate(smoothing_windows):
            fig.add_trace(
                go.Scatter(
                    x=ng.event_data['start_point'],
                    y=ng.event_data[f'snv_{window}'],
                    text=ng.event_data['annotation'],
                    showlegend=False
                ),
                row=value_index + 1,
                col=window_index + 1
            )

    fig.update_layout(height=1200)
    fig.show()


def get_summary_narrativity_heatmap_data(
        event_df: pd.DataFrame,
        narr_col: str,
        minimal_summary_frequency: int = 3) -> pd.DataFrame:

    msf_filtered_df = event_df[event_df['summary_frequency']
                               >= minimal_summary_frequency].copy()
    heatmap_array = {}
    bin_size = (msf_filtered_df[narr_col].max() -
                msf_filtered_df[narr_col].min()) / 25
    bin_iterator = list(
        np.arange(
            event_df[narr_col].min(),
            event_df[narr_col].max(),
            bin_size)
    )
    bin_iterator.reverse()
    for summary_freq in range(minimal_summary_frequency, 9, 1):
        row = {}
        for bin_item in bin_iterator:
            row[round(bin_item + bin_size, 2)] = len(
                event_df[
                    (event_df['summary_frequency'] == summary_freq) &
                    (event_df[narr_col] >= bin_item) &
                    (event_df[narr_col] <= bin_item + bin_size)
                ]
            )
        heatmap_array[summary_freq] = row
    return pd.DataFrame(heatmap_array)


def narr_summary_heatmap(summary_data_file: str):
    smoothing_windows = [20, 100]
    narr_values = [
        [0, 1, 1, 1],
        [0, 1, 2, 3],
        [0, 2, 5, 7],
        [0, 0, 50, 50, ],
    ]
    narr_types = ['non_event', 'stative_event', 'process', 'change_of_state']

    ng = ngc.NarrativityGraph(
        event_annotations=pd.read_json(summary_data_file),
        abs_smoothing_windows=smoothing_windows
    )

    for values in narr_values:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[14, 3])
        narr_dict = dict(zip(narr_types, values))
        ng.compute_narrativity_values(tag_values=narr_dict)
        for smooth_index, window in enumerate(smoothing_windows):

            heatmap_df = get_summary_narrativity_heatmap_data(
                event_df=ng.event_data[int(window/2): - int(window/2)],
                narr_col=f'snv_{window}'
            )
            sns.heatmap(
                heatmap_df,
                annot=False,
                cmap='YlGnBu',
                ax=ax[smooth_index],
                cbar=True,
                cbar_kws={'label': 'Events'},
                # vmax= 150
            )
            ax[smooth_index].set_title(
                f'Smoothing Window: {window}')
            ax[smooth_index].set_xlabel(
                'Summary Frequency')

            if smooth_index == 0:
                ax[smooth_index].set_ylabel('Narrativity Value')
        fig.tight_layout()
        label = str(values).replace('[', '')
        label = label.replace(']', '')
        label = label.replace(', ', '-')
        fig.suptitle(f'Event Type Scaling: {values}')
        fig.show()


def correlation_boxplots(summary_data_df: pd.DataFrame):
    fig = px.box(
        summary_data_df.fillna(0)[
            (summary_data_df['non_event'] == 0) &
            (summary_data_df['smoothing_window'] <= 200) &
            (summary_data_df['minimal_summary_frequency'] <= 3)
        ],
        x='smoothing_window',
        y='correlation',
        hover_data=['non_event', 'stative_event',
                    'process', 'change_of_state'],
        facet_col='minimal_summary_frequency',
        height=800,
        width=1400
    )
    fig.show()
