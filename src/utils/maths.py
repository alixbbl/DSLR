import pandas as pd
import math

class MyMaths():

    def _clean_data(self, data: pd.Series | list) -> list:
        """
        Helper function to clean data regardless of input type.
        
        :param data: Union[pd.Series, List] - Input data
        :return: List - Cleaned list without NaN or None values
        """
        if isinstance(data, pd.Series):
            return data.dropna().tolist()
        else:
            return [x for x in data if x is not None and not (isinstance(x, float) and math.isnan(x)) and isinstance(x, (int, float))]
    
    
    def my_count(self, serie: pd.Series) -> float:
        """"
        This function counts the number of non-null entries in a column.

        :param serie: pd.Series - The column to count non-null entries from.
        :return: int - The count of non-null entries.
        """
        list_non_null = [ele for ele in serie if pd.notnull(ele)]
        return len(list_non_null)

    def my_mean(self, data: pd.Series|list) -> float:
        """
        Calculate the mean of the data.

        :param data: Union[pd.Series, List] - The data to calculate the mean from
        :return: float - The mean of the data
        """
        clean_data = self._clean_data(data)
        if not clean_data:
            return None
        return sum(clean_data) / len(clean_data)

    def my_std(self, serie: pd.Series) -> float:
        """"
        This function returns the standard deviation of a columnn entries.

        :param serie: pd.Series - The column to calculate the standard deviation from.
        :return: float - The standard deviation of the column.
        """
        clean_serie = self._clean_data(serie)
        if not clean_data:
            return None
        n = len(clean_serie)
        mean = self.my_mean(clean_serie)
        var = sum((x - mean) **2 for x in clean_serie) / n
        return math.sqrt(var)

    def my_min(self, data: pd.Series) -> float:
        """
        This function returns the minimum of a columnn entries.

        :param data: pd.Series - The column to calculate the minimum from.
        :return: float - The minimum of the column.
        """
        clean_data = self._clean_data(data)
        if not clean_data:
            return None
        ele_min = clean_data[0]
        for ele in clean_data:
            if ele < ele_min:
                ele_min = ele
        return ele_min

    def my_max(self, data: pd.Series) -> float:
        """
        This function returns the maximum value of a columnn entries.

        :param serie: pd.Series - The column to calculate the maximum from.
        :return: float - The maximum of the column.
        """
        clean_data = self._clean_data(data)
        ele_max = clean_data[0]
        for ele in clean_data:
            if ele > ele_max:
                ele_max = ele
        return ele_max

    def percentile(self, serie: pd.Series, percent: float) -> float:
        """
        This function returns the given percentile of the column's entries.

        :param serie: pd.Series - The column to calculate the percentile from.
        :param percent: float - The percentile to calculate (between 0 and 1).
        :return: float - The calculated percentile of the column.
        """
        sorted_values = serie.dropna().sort_values()
        if sorted_values.empty:
            return None
        n = len(sorted_values)
        index = percent * (n - 1)
        lower = int(index)
        upper = lower + 1 if lower + 1 < n else lower
        weight = index - lower
        return (1 - weight) * sorted_values.iloc[lower] + weight * sorted_values.iloc[upper]

    def my_median(self, serie: pd.Series) -> float:
        """
        Returns the mdian (50 percentiles).
        
        :param serie: pd.Series - The column to calculate the median from.
        :return: float - The median of the column.
        """
        return self.percentile(serie, 0.50)

    def my_25percentile(self, serie: pd.Series) -> float:
        """Returns the 25th percentile.
        
        :param serie: pd.Series - The column to calculate the 25th percentile from.
        :return: float - The 25th percentile of the column.
        """
        return self.percentile(serie, 0.25)

    def my_75percentile(self, serie: pd.Series) -> float:
        """Returns th 75th percentile.
        
        :param serie: pd.Series - The column to calculate the 75th percentile from.
        :return: float - The 75th percentile of the column.
        """
        return self.percentile(serie, 0.75)

    def my_var(self, data):
        """
        Calculate the variance of a dataset.
        
        Variance is the average of squared deviations from the mean.
        Formula: var(X) = (1/n) * Σ(x_i - mean(X))²
        
        :param data: array-like - The dataset to calculate variance for
        :return: float - The variance of the dataset
        """
        data_clean = self._clean_data(data)
        mean = self.my_mean(data_clean)
        sum_squared_diff = sum((x - mean) ** 2 for x in data_clean)
        variance = sum_squared_diff / len(data_clean)
        
        return variance