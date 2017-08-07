import numpy as np
import pandas as pd
import csv


# data macro文件夹存放北京的宏观经济的六个指标 1、国民生产总值 2、商品房平均销售价格 3、社会商品零售总额
#                                            4、居民储蓄年末余额 5、在岗职工平均工资 6、年末总人口
def replace(data):
    for i in range(len(data)):
        data[i] = str(data[i]).replace(',', '')
    return np.array(data).astype(np.float)


def get_macro_data(macro, city):
    macro_value = []
    macro_value.append(macro['DATAN'][(macro['NAME'] == city) & (macro['SJ_CODE'] == 2015)].unique()[0])
    macro_value.append(macro['DATAN'][(macro['NAME'] == city) & (macro['SJ_CODE'] == 2014)].unique()[0])
    macro_value.append(macro['DATAN'][(macro['NAME'] == city) & (macro['SJ_CODE'] == 2013)].unique()[0])
    return replace(macro_value)


def get_sale_data(sale_city, sale_measure):
    # 销量和销额的数据
    sale = pd.read_csv('data/sale/sale.csv', encoding='gbk')
    # 将销售的数据按时间倒叙排列 因为宏观经济是倒叙排序的
    sale.sort_values(by=['COLLECT_PERIOD'], ascending=[0], inplace=True)

    sale_value = sale['MEASURE_VALUE'][
        (sale['SHORT_NAME'] == sale_city) & (sale['MEASURE_NAME'] == sale_measure) &
        (sale['COLLECT_PERIOD'].isin([2013, 2014, 2015]))]
    return replace(list(sale_value))


if __name__ == "__main__":

    saleCityCol = ['北京', '武汉', '长沙', '广州', '深圳', '南宁', '海口', '重庆', '成都', '福州', '厦门', '南昌', '济南', '青岛', '郑州', '银川',
                   '乌鲁木齐',
                   '贵阳', '昆明', '拉萨', '西安', '兰州', '西宁', '呼和浩特', '沈阳', '大连', '长春', '哈尔滨', '上海', '南京', '杭州', '宁波', '合肥',
                   '天津', '石家庄', '太原']
    saleMeasureCol = ['一类烟销量', '二类烟销量', '三类烟销量', '四类烟销量', '五类烟销量', '一类烟销额', '二类烟销额', '三类烟销额', '四类烟销额', '五类烟销额']

    macroMeasureName = ['国民生产总值', '商品房平均销售价格', '社会商品零售总额', '居民储蓄年末余额', '在岗职工平均工资', '年末总人口']
    macros = []
    measure_num = 6

    headers = ['城市', '销售指标', '宏观经济指标', '相关度']
    with open('data/res/result' + '.csv', mode='w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(headers)
        for i in range(1, measure_num + 1):
            macro_data = pd.read_csv('data/macro/' + i.__str__() + '.csv', encoding='gbk')
            for j in range(len(macro_data)):
                macro_data.loc[j, 'NAME'] = macro_data['NAME'][j].replace('市', '')
            macros.append(macro_data)

        for i in range(len(saleCityCol)):
            for j in range(len(saleMeasureCol)):
                array = []
                saleData = get_sale_data(saleCityCol[i], saleMeasureCol[j])
                if len(saleData) == 3:
                    array.append(saleData)
                else:
                    continue
                for k in range(measure_num):
                    macroData = get_macro_data(macros[k], saleCityCol[i])
                    if len(macroData) == 3:
                        array.append(macroData)
                    else:
                        continue
                per_res = np.corrcoef(array)
                for res_index in range(1, len(per_res[0])):
                    res = [saleCityCol[i], saleMeasureCol[j], macroMeasureName[res_index - 1], per_res[0][res_index]]
                    writer.writerow(res)
            print(i)
            # print(np.corrcoef(a))
