import csv
import pandas as pd
from sklearn.cluster import KMeans  # 导入K-means算法包
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


def dict_k(data, random_state):
    dic = []
    k_num = 60
    dict_x = []
    for k in range(1, k_num):
        dict_x.append(k)
        clf_k = KMeans(n_clusters=k)
        clf_k.fit_predict(data)
        dic.append(clf_k.inertia_)
    plt.plot(dict_x, dic)
    plt.xlabel('k值', fontproperties=custom_font)
    plt.ylabel('距离', fontproperties=custom_font)
    plt.show()


def standardization(data, cols, weight):
    for col in range(len(cols)):
        data[cols[col]] = min_max(data[cols[col]]) * weight[col]
    return data


# min_max 标准化函数
def min_max(data):
    min_d = np.min(data)
    max_d = np.max(data)
    return np.array([(dat - min_d) / (max_d - min_d) for dat in data])


if __name__ == "__main__":
    n_clusters = 20
    random_state = 100
    custom_font = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\STFANGSO.TTF')
    #
    itemkv = pd.read_csv('juanyan.csv', encoding='GBK', usecols=[0, 1])
    item_id_index_kv = {}
    item_id = itemkv['ITEM_ID']
    for i in range(len(item_id)):
        item_id_index_kv[item_id[i]] = i

    data = pd.read_csv('juanyan.csv', encoding='GBK', usecols=[2, 3, 4, 5, 7, 8])
    print(type(data))

    # min-max标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    # 需要进行标准化的指标，是否细支没有标准化
    col = data.columns[[0, 1, 2, 4, 5]]

    # 各项指标对应的权重
    weight = [1, 1, 1, 5, 0.3]
    data = standardization(data, col, weight)

    # k-距离图
    # dict_k(data, random_state)

    clf = KMeans(n_clusters=n_clusters, max_iter=500, n_init=200, random_state=random_state)
    y_pred = clf.fit_predict(data)
    print(np.random.mtrand.rand)
    cluster_item_index = []
    for i in range(n_clusters):
        cluster_item_index.append(np.nonzero(y_pred == i)[0])
        if len(cluster_item_index[i]) == 1:
            print('第' + i.__str__() + '类只有一个规格', itemkv['ITEM_NAME'][cluster_item_index[i][0]])
            # sys.exit(0)

    # for i in range(len(y_pred)):
    #     matrix[y_pred[i]].append(i)

    juan_yan = []
    with open('juanyan' + '.csv', mode='r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            juan_yan.append(row)

    # 保存每一类的卷烟名称到result.csv
    with open('result' + '.csv', mode='w', newline='') as wf:
        writer = csv.writer(wf)
        for i in range(n_clusters):
            s1 = '第' + (i + 1).__str__() + '类：'
            # print(s1)
            writer.writerow(['第' + (i + 1).__str__() + '类：'])
            item_name = []
            for j in range(len(cluster_item_index[i])):
                item_name.append(itemkv['ITEM_NAME'][cluster_item_index[i][j]])
            writer.writerow(item_name)
    # 根据聚类结果重排原始数据信息
    juan_yan_len = len((juan_yan[0])) - 1
    with open('julei/result_julei_' + random_state.__str__() + '.csv', mode='w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(headers)
        zong_xiaoliang = []
        for i in range(n_clusters):
            xiaoliang = []
            for j in range(len(cluster_item_index[i])):
                xiaoliang.append(int(juan_yan[cluster_item_index[i][j]][juan_yan_len]))
            zong_xiaoliang.append(sum(xiaoliang))
        xiaoliang_zhanbi = []
        zhanbi_x = []
        for i in range(n_clusters):
            zhanbi_x.append(i + 1)
            xiaoliang_zhanbi.append(float("{:.2f}".format(100 * zong_xiaoliang[i] / sum(zong_xiaoliang))))


        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom')
                # 柱形图边缘用白色填充，纯粹为了美观
                rect.set_edgecolor('white')


        rects = plt.bar(zhanbi_x, xiaoliang_zhanbi, align='center', color='#0072BC', label='类别')
        plt.xticks(zhanbi_x, fontproperties=custom_font)
        plt.title(u'各类总销量占比(%)', fontproperties=custom_font)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5, prop=custom_font)
        plt.xlabel('类别', fontproperties=custom_font)
        plt.ylabel('销量占比(%)', fontproperties=custom_font)
        add_labels(rects)
        outname = "./julei/xiaoliangzhanbi_" + random_state.__str__() + ".png"
        plt.savefig(outname)
        # plt.show()

        for i in range(n_clusters):
            category_information_summary = list(['第' + (i + 1).__str__() + '类：'])
            category_information_summary.append('规格数：' + len(cluster_item_index[i]).__str__())
            category_information_summary.append('销量总数：' + zong_xiaoliang[i].__str__())
            category_information_summary.append('销量占比：' + xiaoliang_zhanbi[i].__str__())

            writer.writerow(category_information_summary)

            # 每类需要求中心点的数据
            category = []
            # print(juan_yan)
            juan_yan_temp = []
            for j in range(len(cluster_item_index[i])):
                category.append(juan_yan[cluster_item_index[i][j]][2:juan_yan_len])
                juan_yan_temp.append(juan_yan[cluster_item_index[i][j]])
                # writer.writerow(juan_yan[matrix[i][j]])

            # 每一组按卷烟销量降序排序写到csv中
            for i in range(len(juan_yan_temp) - 1):
                temp = juan_yan_temp[i]
                index = i
                for j in range(i + 1, len(juan_yan_temp)):
                    if int(juan_yan_temp[j][juan_yan_len]) > int(juan_yan_temp[index][juan_yan_len]):
                        index = j
                if index != i:
                    juan_yan_temp[i] = juan_yan_temp[index]
                    juan_yan_temp[index] = temp
            for i in range(len(juan_yan_temp)):
                writer.writerow(juan_yan_temp[i])
            # print(np.asarray(juan_yan_temp)[:, juan_yan_len])
            category = np.array(category).astype(np.float)
            center = ['中心', '']
            for i in range(category.shape[1]):
                center.append(np.mean(category[:, i]))
            writer.writerow(center)
        writer.writerow([clf.inertia_])
    col = ['m', 'r', 'b', 'c', 'g', 'y', 'black', 'pink', 'aqua', 'tomato', 'olive', 'lightslategray', 'navy',
           'blueviolet',
           'brown', 'chocolate']
    ci = len(col)
    if len(col) > n_clusters:
        ci = n_clusters
    for i in range(ci):
        x = []
        y = []
        for j in range(len(cluster_item_index[i])):
            x.append(juan_yan[cluster_item_index[i][j]][2])
            y.append(juan_yan[cluster_item_index[i][j]][juan_yan_len])
            # x.append(data['TAR_CONT（焦油含量）'][matrix[i][j]])
            # y.append(data['PRICE_TRADE'][matrix[i][j]])
        plt.scatter(x, y, marker='*', color=col[i])
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5, prop=custom_font)
    plt.xlabel('焦油含量', fontproperties=custom_font)
    plt.ylabel('批发价', fontproperties=custom_font)
    # plt.show()
    outname = "./TAR_CONT_price" + ".png"
    plt.savefig(outname)
    # max_price = np.max(data['PRICE_TRADE'])
    # print(max_price)
    # min_price = np.min(data['PRICE_TRADE'])
    # print(min_price)
    # std_price = np.std(data['PRICE_TRADE'])
    # mean_price = np.mean(data['PRICE_TRADE'])
    # pp = (data['PRICE_TRADE'][0] - min_price) / (max_price - min_price)
    # pp1 = (data['PRICE_TRADE'][0] - mean_price) / std_price
    # print(pp, pp1)
    # del_prcie = preprocessing.scale(data['PRICE_TRADE'])
    #
    # print(del_prcie[0])
    # data['PRICE_TRADE'] = data['PRICE_TRADE'] * 0.5


    # Z-score标准化
    # data['PRICE_TRADE'] = preprocessing.scale(data['PRICE_TRADE'])


    print((3 + np.random.rand(4, 1)).flatten())
    print(clf.inertia_)
