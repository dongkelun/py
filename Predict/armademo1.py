# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import csv
import time


class arima_model:
    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxsize

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    # 参数确定模型
    def certain_model(self, p, q):
        model = ARMA(self.data_ts, order=(p, q))
        try:
            self.properModel = model.fit(disp=-1, method='css')
            self.p = p
            self.q = q
            self.bic = self.properModel.bic
            self.predict_ts = self.properModel.predict()
            self.resid_ts = deepcopy(self.properModel.resid)
        except:
            print('You can not fit the model with this parameter p,q, ' \
                  'please use the get_proper_model method to get the best model')

    # 预测第二日的值
    def forecast_next_day_value(self, type='day'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        # self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params
        # print self.properModel.params
        if self.p == 0:  # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        predict_value = np.dot(para[1:], values)
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)


# 差分操作
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts


# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i - 1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i - 1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i - 1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data


from dateutil.relativedelta import relativedelta


def _add_new_data(ts, dat, type='day'):
    if type == 'day':
        new_index = ts.index[-1] + relativedelta(days=1)
    elif type == 'month':
        new_index = ts.index[-1] + relativedelta(months=1)
    ts[new_index] = dat


def add_today_data(model, ts, data, d, type='day'):
    _add_new_data(ts, data, type)  # 为原始序列添加数据
    # 为滞后序列添加新值
    d_ts = diff_ts(ts, d)
    model.add_today_data(d_ts[-1], type)


def forecast_next_day_data(model, d, type='day'):
    if model == None:
        raise ValueError('No model fit before')
    fc = model.forecast_next_day_value(type)
    return predict_diff_recover(fc, d)


def stationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m')
    filename = '2lei'
    index_col = 'DATE1'  # DATE1
    data = pd.read_csv(filename + '.csv', index_col=index_col, date_parser=dateparse)
    # for i in range(len(data['QTY_SOLD_X'])):
    #     data['QTY_SOLD_X'][i]=(float(data['QTY_SOLD_X'][i].replace(',', '')))
    ts = data[data.columns[0]]
    # df = pd.read_csv(filename1, encoding='utf-8', index_col='DATE1')
    # df.index = pd.to_datetime(df.index)
    # ts = df[df.columns[0]]
    # 数据预处理
    mean_n = 12
    rol_mean_s = ts.rolling(window=mean_n).mean()
    ts_log = np.log(ts)

    rol_mean = ts_log.rolling(window=mean_n).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    print(stationarity(ts_diff_1))
    ts_diff_2 = ts_diff_1.diff(2)
    ts_diff_2.dropna(inplace=True)
    ts_diff_3 = ts_diff_2.diff(1)
    ts_diff_3.dropna(inplace=True)

    print(stationarity(ts_diff_2))

    # 模型拟合
    # model = arima_model(ts_diff_2)
    #  这里使用模型参数自动识别
    # model.get_proper_model()
    #
    # print('bic:', model.bic, 'p:', model.p, 'q:', model.q)
    # print(model.properModel.forecast()[0])
    # print(model.forecast_next_day_value(type='month'))
    # # 预测结果还原

    # predict_ts = model.properModel.predict()
    # diff_shift_ts = ts_diff_1.shift(1)
    # diff_recover_1 = predict_ts.add(diff_shift_ts)
    #
    # rol_shift_ts = rol_mean.shift(1)
    # diff_recover = diff_recover_1.add(rol_shift_ts)
    # rol_sum = ts_log.rolling(window=mean_n - 1).sum()
    # rol_recover = diff_recover * mean_n - rol_sum.shift(1)
    # log_recover = np.exp(rol_recover)
    # log_recover.dropna(inplace=True)
    print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccc')

    d = [12, 4]
    diffed_ts = diff_ts(ts_log, d=d)
    print(stationarity(diffed_ts))
    model = arima_model(diffed_ts)
    model.get_proper_model()
    predict_ts = model.properModel.predict()
    print('bic:', model.bic, 'p:', model.p, 'q:', model.q)
    diff_recover_ts = predict_diff_recover(predict_ts, d=d)
    print(np.exp(diff_recover_ts))
    log_recover = np.exp(diff_recover_ts)

    #
    # # 预测结果作图
    import matplotlib as mpl

    custom_font = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\STFANGSO.TTF')
    ts_c = ts[log_recover.index]
    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    ts_c.plot(color='red', label='Original')
    plt.legend(loc='best')
    print((log_recover - ts_c) / ts_c)
    plt.xlabel('日期', fontproperties=custom_font)
    plt.ylabel('销量(条)', fontproperties=custom_font)
    # 均方根误差
    plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - ts_c) ** 2) / ts_c.size))
    outname = './res/' + filename + 'fig1.png'
    plt.savefig(outname)
    # plt.show()

    # ts_train = ts_log[:'2016-04']
    ts_test = ts_log['2016-05':]
    diffed_ts = diff_ts(ts_log, d=d)
    # print('asssssssssssssssssssssss')
    # print(diffed_ts, ts_diff_1 * 12)
    forecast_list = []
    # for i, dta in enumerate(ts_test):
    for i in range(8):
        # if i % 7 == 0:
        #     model = arima_model(diffed_ts)
        #     model.get_proper_model()
        forecast_data = forecast_next_day_data(model, d=d, type='month')
        forecast_list.append(forecast_data)
        add_today_data(model, ts_log, forecast_data, d, type='month')
        _add_new_data(ts_test, None, 'month')
    predict_ts = pd.Series(data=forecast_list, index=ts_test['2017-05':].index)
    log_recover = np.exp(predict_ts)
    print(log_recover)
    res = ['p:' + model.p.__str__(), 'q:' + model.q.__str__()]
    with open('./res/' + filename + '_predict.csv', mode='w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(res)

        ts_2017 = ts['2017-01':].append(log_recover)

        cou = 0
        ts_2017_arr = np.array(ts_2017)
        for index_i in ts_2017.index:
            ts_date = str(index_i)[:7]
            writer.writerow([ts_date.replace('-', ''), round(ts_2017_arr[cou], 2)])
            cou += 1
        writer.writerow(['总量', round(sum(ts['2017-01':]) + sum(log_recover), 2)])
    original_ts = ts.append(log_recover)

    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    original_ts[:'2017-06'].plot(color='red', label='Original')
    plt.xlabel('日期', fontproperties=custom_font)
    plt.ylabel('销量(条)', fontproperties=custom_font)
    plt.legend(loc='best')
    outname = './res/' + filename + 'fig2.png'
    plt.savefig(outname)
    # plt.show()
