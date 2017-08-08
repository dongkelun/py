import numpy as np
import pandas as pd
import csv


def cal_euler_dist(arr_a, arr_b):
    """
    功能：欧拉距离计算
    输入：两个一维数组
    """
    return np.math.sqrt(sum(np.power(np.array(arr_a) - np.array(arr_b), 2)))


dataCol = '相关度'
cityCol = '城市'
person_data = pd.read_csv('data/res/result.csv', encoding='gbk')
saleCityCol = ['北京', '武汉', '长沙', '广州', '深圳', '南宁', '海口', '重庆', '成都', '福州', '厦门', '南昌', '济南', '青岛', '郑州', '银川',
               '乌鲁木齐', '贵阳', '昆明', '拉萨', '西安', '兰州', '西宁', '呼和浩特', '沈阳', '大连', '长春', '哈尔滨', '上海', '南京', '杭州', '宁波',
               '合肥', '天津', '石家庄', '太原']
dic = []
for i in range(1, len(saleCityCol)):
    dic.append(cal_euler_dist(person_data[dataCol][person_data[cityCol] == saleCityCol[0]],
                              person_data[dataCol][person_data[cityCol] == saleCityCol[i]]))
for i in range(len(dic)):
    if np.isnan(dic[i]):
        dic[i] = np.inf
index = np.argsort(dic)
headers = ['城市', '距离']
with open('data/res/result_similar' + '.csv', mode='w', newline='') as wf:
    writer = csv.writer(wf)
    writer.writerow(headers)
    for i in range(len(index)):
        writer.writerow([saleCityCol[index[i] + 1], dic[index[i]]])
print(sorted(dic))
