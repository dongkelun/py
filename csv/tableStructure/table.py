# 表结构整理

import pandas as pd
import re
import math
import numpy as np

f1 = open('table.txt', 'w')
data = pd.read_csv('table.csv', encoding='gbk')
col = ['字段编码', '字段名称', '数据类型', '备注']
s1 = "<column id=\""
for i in range(len(data)):
    s = s1
    if isinstance(data[col[3]][i], str) and bool(re.search(data[col[3]][i], '主键', re.IGNORECASE)):
        s += data[col[0]][i].strip().upper() + "\" required=\"true\" primaryKey=\"true\""
    else:
        s += data[col[0]][i].strip().upper() + "\" required=\"false\" primaryKey=\"false\""
    col2 = data[col[2]][i].replace(' ', '')
    if bool(re.search('VARCHAR', col2, re.IGNORECASE)):
        s += " type=\"VARCHAR\" size=\"" + col2[8:len(col2) - 1] + "\""
    elif bool(re.search('CHAR', col2, re.IGNORECASE)):
        s += " type=\"CHAR\" size=\"" + col2[5:len(col2) - 1] + "\""
    elif bool(re.search('NUMERIC', col2, re.IGNORECASE)):
        s += " type=\"DECIMAL(" + col2[8:len(col2)] + "\""
    elif bool(re.search('INTEGER', col2, re.IGNORECASE)):
        s += " type=\"INTEGER\""
    elif bool(re.search('DATE', col2, re.IGNORECASE)):
        s += " type=\"DATE\""
    if isinstance(data[col[3]][i], str) and not bool(re.search(data[col[3]][i], '主键', re.IGNORECASE)):
        s += " name=\"" + data[col[1]][i].strip() + "\" enumValue=\"" + data[col[3]][i].strip() + "\" />"
    else:
        s += " name=\"" + data[col[1]][i].strip() + "\" />"
    print(s)
    f1.write(s + '\n')



# insert_sql = 'INSERT INTO DB2INST2.DL_BLOCK_CUST\
# (BLOCK_ID, CUST_ID) VALUES('
# for j in range(len(data)):
#     sql = insert_sql
#
#     if data[col[0]][j] !=data[col[0]][j]:
#         continue
#     sql += "'" + data[col[0]][j] + "',"
#     for i in range(1, len(col) - 1):
#         sql += "'" + data[col[i]][j].__str__() + "',"
#
#     sql += "'" + data[col[len(col) - 1]][j].__str__() + "');"
#     print(sql)
#     f1.write(sql + '\n')
