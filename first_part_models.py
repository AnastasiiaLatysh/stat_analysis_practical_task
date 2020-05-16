from statsmodels.tsa.arima_model import ARIMA

from common_calculations import get_the_stationary_series
from first_part_calculations import show_statistical_data, ma_by_ar_resid

# отримання стаціонарного часового ряду
stationary_series = get_the_stationary_series()

# АРКС(2)
arks_model = ARIMA(stationary_series, order=(2, 0, 0))
model = arks_model.fit(disp=0)
AR_resid = model.resid
split = len(stationary_series) - int(0.2 * len(stationary_series))
train, test = stationary_series[0:split], stationary_series[split:]
pred = model.predict(len(test))
show_statistical_data(train, model, pred)

ma_by_ar_resid(AR_resid, 2, 0, 4, 'АРКС(2,4)')

arks_n5_PKC = AR_resid.rolling(5).mean().fillna(AR_resid[:5].mean())
ma_by_ar_resid(arks_n5_PKC, 2, 0, 3, 'АРКС(2,3) із застосуванням власного простого КС, при N=5')

arks_n10_PKC = AR_resid.rolling(10).mean().fillna(AR_resid[:10].mean())
ma_by_ar_resid(arks_n10_PKC, 2, 0, 7, 'АРКС(2,7) із застосуванням власного простого КС, при N=10')

arks_n5_EKC = AR_resid.ewm(5).mean()
ma_by_ar_resid(arks_n5_EKC, 2, 0, 3, 'АРКС(2,3) із застосуванням власного експоненційного КС, при N=5')

arks_n10_EKC = AR_resid.ewm(10).mean()
ma_by_ar_resid(arks_n10_EKC, 2, 0, 6, 'АРКС(2,6) із застосуванням власного експоненційного КС, при N=10')
