import matplotlib.pyplot as plt
import numpy
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

from common_calculations import show_statistical_data, show_dfuller_statistic, LAGS

pd.plotting.register_matplotlib_converters()


# АРКС(p,q) де КС побудоване по залишкам АР(p) рівняння
def ma_by_ar_resid(data, p=0, d=0, q=0, model_name=''):
    # розрахунок автокореляції та вивидення графіку
    plot_acf(data, lags=LAGS)
    plt.show()

    # подубова моделі авторегресії ковзного середнього
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(disp=0)
    split = len(data) - int(0.2 * len(data))
    train, test = data[0:split], data[split:]
    pred = model_fit.predict(len(test))
    print(f'\n------ {model_name} ------')
    show_statistical_data(train, model_fit, pred)
