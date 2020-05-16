import matplotlib.pyplot as plt
import numpy
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

LAGS = 12


def show_statistical_data(data, model_fit, pred):
    # SSres = SUM((y - pred(x))^2)
    SSres = numpy.sum((data - pred)**2)
    avg_y = (1 / len(data)) * numpy.sum(data)
    # SStot = SUM( (y - avg(y))^2 )
    SStot = numpy.sum((data - avg_y)**2)
    r_sq = 1 - SSres / SStot
    print(f"R — squared: {r_sq}")
    print(f"Sum squared resid: {SSres}")
    print(f"Durbin — Watson stat: {sm.stats.durbin_watson(model_fit.resid.values)}")
    print(f'{model_fit.summary()} \n\n')


def show_dfuller_statistic(data):
    result = adfuller(data)
    print(f'ADF Statistics: {result[0]}\n')
    print(f'p-value: {result[1]}\n')
    print('Critical Values:\n')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


# приведення ряду до стаціонарного
def get_the_stationary_series():
    df = pd.read_csv("1997rts1.txt", sep="\n", decimal=".")
    print('Перевірки часового ряду на стаціонарність – розширений тест Дікі–Фулера')
    show_dfuller_statistic(df)

    df_log = numpy.log(df)
    moving_avg = df_log.rolling(LAGS).mean().fillna(df_log[:LAGS].mean())
    df_log_moving_avg_diff = df_log - moving_avg

    print('Аналогічний Дікі–Фулер тест, але вже для ряду в перших різницях')
    show_dfuller_statistic(df_log_moving_avg_diff)

    plot_pacf(df_log_moving_avg_diff, lags=LAGS)
    plt.show()
    return df_log_moving_avg_diff
