from typing import Generator
import random
import numpy as np
from scipy import signal
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def filter_ac_by_start_point(df1, df2, span: int = 1) -> pd.DataFrame:
    compare_start_points1 = []
    for sp in df1['start_point']:
        compare_start_points1.extend(list(range(sp - span, sp + span + 1, 1)))

    compare_start_points2 = []
    for sp in df2['start_point']:
        compare_start_points2.extend(list(range(sp - span, sp + span + 1, 1)))

    filtered_df1 = df1[df1['start_point'].isin(compare_start_points2)].copy()
    filtered_df2 = df2[df2['start_point'].isin(compare_start_points1)].copy()

    return filtered_df1, filtered_df2


def narr_values_generator(
        number_of_values: int = 10,
        maximal_value: int = 100) -> pd.DataFrame:
    """
    Generate Random Narrtivity Values for Event Types.

    Args:
        number_of_values (int, optional): number of narr values. Defaults to 10.
        maximal_value (int, optional): highest narr value.

    Returns:
        pd.DataFrame: [description]
    """
    random_lists = []
    while number_of_values > len(random_lists):
        randomlist = sorted([
            random.randint(0, maximal_value) for i in range(0, 4)
        ])

        if randomlist not in random_lists and min(randomlist) == 0 and randomlist[2] > 0:
            random_lists.append(randomlist)

    return random_lists


def peak_text(event_df: pd.DataFrame, peak_start: int, peak_end: int):
    peak_text = '<br>'
    for index, row in event_df[
        (event_df['start_point'] >= peak_start) &
        (event_df['end_point'] <= peak_end)
    ].iterrows():
        peak_text += row['annotation']
        peak_text += '<br>'

    return peak_text


def smooth_timeline(timeline: list, window_size: int = 10):
    if window_size % 2 == 0:
        window_size += 1
    window = signal.windows.cosine(window_size)
    mid_window = int(window_size / 2)

    for index, _ in enumerate(timeline):
        left_pointer = index - mid_window
        right_pointer = index + mid_window + 1

        left_values = timeline[left_pointer: index]
        right_values = timeline[index: right_pointer]

        left_smoothing_window = window[0: mid_window]
        right_smoothing_window = window[mid_window: mid_window +
                                        len(right_values) + 1]

        if left_pointer < 0:
            left_pointer = 0
            left_values = timeline[left_pointer: index]
            left_smoothing_window = window[mid_window -
                                           len(left_values): mid_window]

        if right_pointer > len(timeline):
            right_pointer = len(timeline)
            right_values = timeline[index: right_pointer]
            right_smoothing_window = window[mid_window: mid_window +
                                            len(right_values)]

        left_smoothing_sum = sum(left_smoothing_window)
        right_smoothing_sum = sum(right_smoothing_window)

        if len(left_values) == 0:
            right_smoothed_values = sum(
                np.array(right_values) * right_smoothing_window
            ) / right_smoothing_sum
            smooth_value = np.mean(right_smoothed_values)
        else:
            left_smoothed_values = sum(
                np.array(left_values) * left_smoothing_window
            ) / left_smoothing_sum
            right_smoothed_values = sum(
                np.array(right_values) * right_smoothing_window
            ) / right_smoothing_sum
            smooth_value = np.mean(
                [left_smoothed_values, right_smoothed_values])

        yield smooth_value


class NarrativityGraph:
    def __init__(
            self,
            event_annotations: pd.DataFrame,
            abs_smoothing_windows: list = None) -> None:
        self.default_tag_values = {
            'non_event': 0,
            'stative_event': 2,
            'process': 5,
            'change_of_state': 7
        }
        self.default_property_values = {
            'prop:iterative': {
                'yes': 0,
                'no': -2
            },
            'prop:representation_type': {
                'narrator_speech': 0,
                'speech_representation': -2,
                'thought_representation': -2,
            }
        }

        self.event_data = event_annotations[
            event_annotations['tag'].isin(list(self.default_tag_values))
        ].copy()

        if abs_smoothing_windows:
            self.abs_smoothing_window = abs_smoothing_windows
        else:
            self.abs_smoothing_window = [30, 50, 100, 150, 200, 300, 500]

        self.rel_smoothing_window = [0.005, 0.01, 0.02, 0.05, 0.01]

    def compute_narrativity_values(
            self, tag_col: str = 'tag',
            abs_smoothing_windows: list = None,
            rel_smoothing_windows: list = None,
            abs_smoothing: bool = True,
            tag_values: dict = None,
            prop_values: dict = None,
            include_props: bool = False,) -> None:
        """Computes Narrativity Values using the `NarrativityGraph.event_data` DataFrame.

        Args:
            tag_col (str, optional): The column in the `self.event_data` with the event types. Defaults to 'tag'.
            abs_smoothing_windows (list, optional): Absoulute smoothing windows. Defaults to None.
            rel_smoothing_windows (list, optional): Smoothing window sizes relative to the text's length. Defaults to None.
            abs_smoothing (bool, optional): Whether absolute or relative smoothing should be used. Defaults to True.
            tag_values (dict, optional): A dictionary with event types as keys and integers as values. Defaults to None.
            prop_values (dict, optional): A dictionary with event properties as keys and integers as values. Defaults to None.
            include_props (bool, optional):  Whether event properties should be used. Defaults to False.
        """
        if not tag_values:
            tag_values = self.default_tag_values
        if not prop_values:
            prop_values = self.default_property_values
        if not abs_smoothing_windows:
            abs_smoothing_windows = self.abs_smoothing_window
        if not rel_smoothing_windows:
            rel_smoothing_windows = self.rel_smoothing_window
        else:
            abs_smoothing = False

        # get narr value for each row in event data
        if include_props:
            snv = []
            for _, row in self.event_data.iterrows():
                prop_value = sum([
                    prop_values[row[f'prop:{prop}']] for prop in prop_values
                ])
                row_value = row['tag_narr_values'] + prop_value
                snv.append(row_value)
            self.event_data.loc[:, 'narr_value'] = snv
        else:
            self.event_data.loc[:, 'narr_values'] = [
                tag_values[row[tag_col]] for _, row in self.event_data.iterrows()
            ]

        # get smooth narr value for each row in event data
        if abs_smoothing:
            for smoothing_window in abs_smoothing_windows:      # iterate over different smoothing windows
                self.event_data.loc[:, f'snv_{smoothing_window}'] = self.event_data['narr_values'].rolling(
                    window=smoothing_window, center=True, win_type='cosine'
                ).mean()

        else:
            for smoothing_window in rel_smoothing_windows:
                window = int(smoothing_window * len(self.event_data))
                self.event_data.loc[:, f'snv_{smoothing_window}'] = self.event_data['narr_values'].rolling(
                    window=window, center=True, win_type='cosine'
                ).mean()
