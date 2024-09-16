import pandas as pd


class Utils:

    @staticmethod
    def missing_values_percentage(dataframe: pd.DataFrame) -> None:
        complete_rows_df_level = ((dataframe.notna().all(axis=1).sum() / dataframe.shape[0]) * 100).round(2)
        print(f'There are {complete_rows_df_level}% complete rows in this dataset.')

        missing_values = ((dataframe.isna().sum() / dataframe.shape[0]) * 100).sort_values(ascending=False).round(2)
        print(f'Missing values in percentage:\n{missing_values[missing_values > 0]}%')
