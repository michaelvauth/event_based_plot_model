import pandas as pd
import re
import ast

from pandas.core.frame import DataFrame


def findoverlap(
        an1_start_point: int,
        an1_end_point: int,
        an2_start_point: int,
        an2_end_point: int,) -> None:
    """
    Test if annotation 2 starts or ends within annotations 1 span.
    Args:
        an1_start_point (int): [description]
        an1_end_point (int): [description]
        an2_start_point (int): [description]
        an2_end_point (int): [description]
    Returns:
        [type]: [description]
    """
    # test if an2 starts before or in an1 and ends in an1
    if an1_start_point <= an2_start_point < an1_end_point:
        return True
    # test if an2 ends in or with an1
    elif an1_start_point < an2_end_point <= an1_end_point:
        return True
    # test if an2 starts before and ends after an1 -> includes an1
    elif an2_start_point < an1_start_point and an2_end_point > an1_end_point:
        return True
    else:
        return False


def shorter_ids(id_label):
    if '_' in id_label:
        return re.findall(r'(.*?)_', id_label)[0]
    else:
        return id_label


def get_summary_value_for_event(
        event_df: pd.DataFrame,
        summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_values = []
    labels = []
    events_per_summary = {}
    for _, row in event_df.iterrows():
        filtered_df = summary_df[
            [
                findoverlap(
                    an1_start_point=row['start_point'],
                    an1_end_point=row['end_point'],
                    an2_start_point=summary_row['start_point'],
                    an2_end_point=summary_row['end_point']
                ) for _, summary_row in summary_df.iterrows()
            ]
        ].copy().reset_index(drop=True)
        filtered_labels = [item[0]
                           for item in filtered_df['prop:summary_index']]
        labels.append(filtered_labels)
        summary_values.append(
            len(set([shorter_ids(item) for item in filtered_labels]))
        )
        for label in labels[-1]:
            short_label = shorter_ids(label)
            if short_label in events_per_summary:
                events_per_summary[short_label] += 1
            else:
                events_per_summary[short_label] = 1

    event_df.loc[:, 'summary_frequency'] = summary_values
    event_df.loc[:, 'summary_ids'] = labels

    # normalize the summary frequency by summaries per student
    normalized_summary_values = []
    for _, row in event_df.iterrows():
        norm_row_value = 0
        for item in row['summary_ids']:
            norm_row_value += 1 / events_per_summary[
                shorter_ids(item)]
        normalized_summary_values.append(norm_row_value)
    event_df.loc[:, 'normalized_summary_freq_by_student'] = normalized_summary_values

    return event_df


class Configuration:
    def __init__(self, configuration_str: str) -> None:
        list_repr = ast.literal_eval(configuration_str)
        self.non_event = list_repr[0]
        self.stative_event = list_repr[1]
        self.process = list_repr[2]
        self.change_of_state = list_repr[3]
        self.smoothing_window = list_repr[4]
        self.minimal_summary_frequency = list_repr[5]

    def filter_df(self, df: pd.DataFrame):
        return df[
            (df.non_event == self.non_event) &
            (df.stative_event == self.stative_event) &
            (df.process == self.process) &
            (df.change_of_state == self.change_of_state) &
            (df.smoothing_window == self.smoothing_window) &
            (df.minimal_summary_frequency == self.minimal_summary_frequency)
        ].correlation.mean()


def get_mean_value_per_configuration(summary_data_df) -> pd.DataFrame:
    configurations = set(
        [
            str(item) for item in zip(
                summary_data_df['non_event'],
                summary_data_df['stative_event'],
                summary_data_df['process'],
                summary_data_df['change_of_state'],
                summary_data_df['smoothing_window'],
                summary_data_df['minimal_summary_frequency'],
            )
        ]
    )

    mean_corr_data = []
    for conf in configurations:
        configuration = Configuration(conf)
        config_dict = configuration.__dict__
        config_dict['correlation mean'] = configuration.filter_df(
            df=summary_data_df)
        mean_corr_data.append(config_dict)

    return pd.DataFrame(mean_corr_data)


def get_best_configurations(mean_corr_df: pd.DataFrame):
    mean_df = pd.DataFrame(mean_corr_df)
    mean_df['correlation mean'] = mean_df['correlation mean'].round(4)

    for item in range(4):
        filtered_mean_df = mean_df[
            mean_df.minimal_summary_frequency == item
        ]
        filtered_mean_df.drop(
            ['minimal_summary_frequency'], axis=1, inplace=True)
        sorted_df = filtered_mean_df.sort_values(
            'correlation mean', ascending=False).head(5)
        return sorted_df
