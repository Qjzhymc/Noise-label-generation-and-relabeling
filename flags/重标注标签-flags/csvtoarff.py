"""将csv文件转换成为arff文件"""
def csv2arff(fpath):
    import pandas as pd
    df=pd.read_csv(fpath)
    columns=df.columns.tolist()
    datatype=["numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric"]
    fpath=fpath[:fpath.find('.csv')]+'1.arff'
    f = open(fpath,'w+') 
    f.write('@relation {}\n'.format('airline_passengers'))
    for i in range(len(columns)):
        f.write('@attribute {} {}\n'.format(columns[i],datatype[i]))
    f.write('@data\n')
    for i in range(df.shape[0]):
        item=df.iloc[i,:df.shape[1]]
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(item[0],item[1],item[2],
                item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11]
                ,item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20]
                ,item[21],item[22],item[23],item[24],item[25]))
    f.close()
fpath='C:/Users/22467/Desktop/噪声多标签/实验/flags/重标注标签-flags/flags_0.9.csv'
csv2arff(fpath)
