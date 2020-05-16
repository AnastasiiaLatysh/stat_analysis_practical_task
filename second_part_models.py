from statsmodels.tsa.arima_model import ARIMA

from common_calculations import get_the_stationary_series
from second_part import ma_by_y

# отримання стаціонарного часового ряду
stationary_series = get_the_stationary_series()

model = ARIMA(stationary_series, order=(2, 0, 0))
model_fit = model.fit(disp=0)
ar_fitval = model_fit.fittedvalues

sma_5 = ar_fitval.rolling(5).mean().fillna(ar_fitval[:5].mean())
ma_by_y(sma_5, 2, 0, 4, 'АРКС(2,4) із застосуванням власного простого КС по у, при N=5')

sma_10 = ar_fitval.rolling(10).mean().fillna(ar_fitval[:10].mean())
ma_by_y(sma_10, 2, 0, 5, 'АРКС(2,5) із застосуванням власного простого КС по у, при N=10')

ema_5 = ar_fitval.ewm(5).mean()
ma_by_y(ema_5, 2, 0, 6, 'АРКС(2,6) із застосуванням власного експоненційного КС по у, при N=5')

ema_10 = ar_fitval.ewm(10).mean()
ma_by_y(ema_10, 2, 0, 7, 'АРКС(2,7) із застосуванням власного експоненційного КС по у, при N=10')
