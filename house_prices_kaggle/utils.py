import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Utils:

    @staticmethod
    def missing_values_percentage(dataframe: pd.DataFrame) -> None:
        """
        Function to calculate the percentage of complete rows in dataframe and percentage of missing values in specific
        column
        :param dataframe: Dataframe
        :return:
        """
        complete_rows_df_level = ((dataframe.notna().all(axis=1).sum() / dataframe.shape[0]) * 100).round(2)
        print(f'There are {complete_rows_df_level}% complete rows in this dataset.')

        missing_values = ((dataframe.isna().sum() / dataframe.shape[0]) * 100).sort_values(ascending=False).round(2)
        print(f'Missing values in percentage:\n{missing_values[missing_values > 0]}%')

    @staticmethod
    def detect_iqr_outliers(dataframe: pd.DataFrame, column: pd.DataFrame.columns) -> int:
        """
        Function for detecting outliers using IQR method.
        :param dataframe: Dataframe
        :param column: Column for outlier detection
        :return: Count of outliers
        """
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)

        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outliers_count = len(outliers)

        return outliers_count

    @staticmethod
    def create_hist_plot(df: pd.DataFrame, column: pd.DataFrame.columns) -> None:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, stat='count')
        plt.title(f'Histogram of {column}')
        plt.show()