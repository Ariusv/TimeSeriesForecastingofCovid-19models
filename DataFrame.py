import pandas as pd
from config import  *
import numpy as np
class GeneralCovidDataFrame:
    def __init__(self):
        self.confirmed_cases_df = pd.read_csv(LINK_TO_CONFIRMED_CASES_REPOSITORY)
        self.deaths_df = pd.read_csv(LINK_TO_DEATHS_REPOSITORY)
        self.recoveries_df = pd.read_csv(LINK_TO_RECOVERIES_REPOSITORY)
        self.columns = self.confirmed_cases_df.keys()
        self.all_dates = self.confirmed_cases_df.loc[:, self.columns[4]:self.columns[-1]].keys()


class CountryCovidDataFrame:
    def __init__ (self, covid_dataframe, country_name):
        self.country_name = country_name
        self.__covid_dataframe = covid_dataframe
        self.dates = np.array(covid_dataframe.all_dates)
        self.__extract_covid_data()
        self.__create_df()

    def __extract_covid_data(self):
        def daily_increase(data):
            d = []
            for i in range(len(data)):
                if i == 0:
                    d.append(data[0])
                else:
                    d.append(data[i] - data[i - 1])
            return d

        self.confirmed_cases = []
        self.deaths = []
        self.recoveries = []

        for i in self.__covid_dataframe.all_dates:
            self.confirmed_cases.append(self.__covid_dataframe.confirmed_cases_df[self.__covid_dataframe.confirmed_cases_df['Country/Region']==self.country_name][i].sum())
            self.deaths.append(self.__covid_dataframe.deaths_df[self.__covid_dataframe.deaths_df['Country/Region']==self.country_name][i].sum())
            self.recoveries.append(self.__covid_dataframe.recoveries_df[self.__covid_dataframe.recoveries_df['Country/Region']==self.country_name][i].sum())

        self.daily_confirmed_cases = daily_increase(self.confirmed_cases)
        self.daily_deaths = daily_increase(self.deaths)
        self.daily_recoveries = daily_increase(self.recoveries)
    def __create_df(self):
        grouped_confirmed_cases = self.__covid_dataframe.confirmed_cases_df.groupby('Country/Region')
        grouped_recovered_df = self.__covid_dataframe.deaths_df.groupby('Country/Region')
        grouped_death_df = self.__covid_dataframe.recoveries_df.groupby('Country/Region')
        country_confirmed_df = grouped_confirmed_cases.get_group(self.country_name)
        country_recovered_df = grouped_recovered_df.get_group(self.country_name)
        country_death_df = grouped_death_df.get_group(self.country_name)
        cleaned_confirmed_df = country_confirmed_df.drop(columns=['Province/State', 'Lat', 'Long']).sum(axis=0, skipna=True).to_frame()[1:]
        cleaned_recovered_df = country_recovered_df.drop(columns=['Province/State', 'Lat', 'Long']).sum(axis=0, skipna=True).to_frame()[1:]
        cleaned_death_df = country_death_df.drop(columns=['Province/State', 'Lat', 'Long']).sum(axis=0,skipna=True).to_frame()[1:]
        self.df_for_learning = pd.DataFrame(cleaned_confirmed_df.values, columns=['confirmed_cases'])
        self.df_for_learning['recoveries'] = cleaned_recovered_df.values
        self.df_for_learning['deaths'] = cleaned_death_df.values
        self.df_for_learning.index = pd.to_datetime(cleaned_confirmed_df.index)
        self.df_for_learning['daily_confirmed_cases'] = self.df_for_learning.confirmed_cases.diff().dropna().astype(np.int64)
        self.df_for_learning['daily_recoveries'] = self.df_for_learning.recoveries.diff().dropna().astype(np.int64)
        self.df_for_learning['daily_deaths'] = self.df_for_learning.deaths.diff().dropna().astype(np.int64)
        self.df_for_learning.iloc[0] = 0

class WorldCovidDataFrame:
    def __init__ (self, covid_dataframe):
        self.country_name = "World"
        self.__covid_dataframe = covid_dataframe
        self.dates = covid_dataframe.all_dates
        self.__extract_covid_data()

    def __extract_covid_data(self):
        def daily_increase(data):
            d = []
            for i in range(len(data)):
                if i == 0:
                    d.append(data[0])
                else:
                    d.append(data[i] - data[i - 1])
            return d

        self.confirmed_cases = []
        self.deaths = []
        self.recoveries = []
        columns = self.__covid_dataframe.columns

        all_confirmed_cases = self.__covid_dataframe.confirmed_cases_df.loc[:, columns[4]:columns[-1]]
        all_deaths = self.__covid_dataframe.deaths_df.loc[:, columns[4]:columns[-1]]
        all_recoveries = self.__covid_dataframe.recoveries_df.loc[:, columns[4]:columns[-1]]

        for i in self.__covid_dataframe.all_dates:
            self.confirmed_cases.append(all_confirmed_cases[i].sum())
            self.deaths.append(all_deaths[i].sum())
            self.recoveries.append(all_recoveries[i].sum())



        self.daily_confirmed_cases = daily_increase(self.confirmed_cases)
        self.daily_deaths = daily_increase(self.deaths)
        self.daily_recoveries = daily_increase(self.recoveries)





