﻿"""将csv文件转换成为arff文件"""
def csv2arff(fpath):
    import pandas as pd
    df=pd.read_csv(fpath)
    columns=df.columns.tolist()
    datatype=["numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric"
    ,"numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric"
    ,"numeric","numeric","numeric","numeric","numeric","numeric"]
    fpath=fpath[:fpath.find('.csv')]+'1.arff'
    f = open(fpath,'w+') 
    f.write('@relation {}\n'.format('airline_passengers'))
    for i in range(len(columns)):
        f.write('@attribute {} {}\n'.format(columns[i],datatype[i]))
    f.write('@data\n')
    for i in range(df.shape[0]):
        item=df.iloc[i,:df.shape[1]]
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(item[0],item[1],item[2],
                item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11]
                ,item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20]
                ,item[21],item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29]
                ,item[30],item[31],item[32],item[33],item[34],item[35],item[36],item[37]
                ,item[38],item[39],item[40],item[41],item[42],item[43],item[44],item[45],item[46]
                ,item[47],item[48],item[49],item[50],item[51],item[52],item[53],item[54],item[55]
                ,item[56],item[57],item[58],item[59],item[60],item[61],item[62],item[63]
                ,item[64],item[65],item[66],item[67],item[68],item[69],item[70],item[71],item[72]
                ,item[73],item[74],item[75],item[76],item[77],item[78],item[79],item[80],item[81]
                ,item[82],item[83],item[84],item[85],item[86],item[87],item[88],item[89],item[90]
                ,item[91],item[92],item[93],item[94],item[95],item[96],item[97],item[98],item[99]
                ,item[100],item[101],item[102],item[103],item[104],item[105],item[106],item[107],item[108]
                ,item[109],item[110],item[111],item[112],item[113],item[114],item[115],item[116]))
    f.close()
fpath='C:/Users/22467/Desktop/噪声多标签/实验/yeast/yeast重标注数据/yeast_0.9.csv'
csv2arff(fpath)
