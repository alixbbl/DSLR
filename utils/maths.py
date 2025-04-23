import pandas as pd
import math

class MyMaths():
    
    def my_count(self, serie: pd.Series) -> float:
        """"
        This function counts the number of non-null entries in a column.

        :param serie: pd.Series - The column to count non-null entries from.
        :return: int - The count of non-null entries.
        """
        list_non_null = [ele for ele in serie if pd.notnull(ele)]
        return len(list_non_null)

    def my_mean(self, serie: pd.Series) -> float:
        """
        This function returns the means of a columnn entries.

        :param serie: pd.Series - The column to calculate the mean from.
        :return: float - The mean of the column.
        """
        clean_serie = serie.dropna()
        if clean_serie.empty:
            return None
        return sum(clean_serie) / len(clean_serie) if len(clean_serie) > 0 else None

    def my_std(self, serie: pd.Series) -> float:
        """"
        This function returns the standard deviation of a columnn entries.

        :param serie: pd.Series - The column to calculate the standard deviation from.
        :return: float - The standard deviation of the column.
        """
        clean_serie = serie.dropna()
        if clean_serie.empty:
            return None
        n = len(clean_serie)
        mean = self.my_mean(clean_serie)
        var = sum((x - mean) **2 for x in clean_serie) / n
        return math.sqrt(var)

    def my_min(self, serie: pd.Series) -> float:
        """
        This function returns the minimum of a columnn entries.

        :param serie: pd.Series - The column to calculate the minimum from.
        :return: float - The minimum of the column.
        """
        clean_serie=serie.dropna()
        if clean_serie.empty:
            return None
        ele_min = clean_serie[0]
        for ele in clean_serie:
            if ele < ele_min:
                ele_min = ele
        return ele_min

    def my_max(self, serie: pd.Series) -> float:
        """
        This function returns the maximum value of a columnn entries.

        :param serie: pd.Series - The column to calculate the maximum from.
        :return: float - The maximum of the column.
        """
        clean_serie=serie.dropna()
        if clean_serie.empty:
            return None
        ele_max = clean_serie[0]
        for ele in clean_serie:
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
        sorted_values = serie.dropna().sort_values().to_numpy() # on clean si jamais
        n = len(sorted_values)
        if n == 0:
            return None
        index = percent * (n - 1)
        lower = int(index)
        upper = lower + 1 if lower + 1 < n else lower
        weight = index - lower
        return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]

    def my_median(self, serie: pd.Series) -> float:
        """
        Returns the mdian (50 percentiles).
        
        :param serie: pd.Series - The column to calculate the median from.
        :return: float - The median of the column.
        """
        return self.percentile(serie, 0.50)

    def my_25percentile(self, serie: pd.Series) -> float:
        """Returns the 25th percentile."""
        return self.percentile(serie, 0.25)

    def my_75percentile(self, serie: pd.Series) -> float:
        """Returns th 75th percentile."""
        return self.percentile(serie, 0.75)
