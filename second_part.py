import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

from common_calculations import show_statistical_data, LAGS

pd.plotting.register_matplotlib_converters()


# Побудова АРКС(p,q) де КС побудоване по вихідному сигналу у
def ma_by_y(data, p=0, d=0, q=0, model_name=''):
    plot_acf(data, lags=LAGS)
    plt.show()

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(disp=0)

    split = len(data) - int(0.2*len(data))
    train, test = data[0:split], data[split:]
    pred = model_fit.predict(len(test))

    print('\n------ ', model_name, ' ------')
    show_statistical_data(train, model_fit, pred)
