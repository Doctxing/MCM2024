import pandas as pd
from pandas import Series, DataFrame


Db_B = pd.read_csv('BCHAIN-MKPRU.csv')
Db_G = pd.read_csv('LBMA-GOLD.csv')

def Generation(Db_B,Db_G):
    Db = pd.DataFrame(columns=['Date','Btdy','Bpdt','Gtdy','Gpdt','Call','Crst','Ball','Brst','Gall','Grst','Rate'])
    for row in range(0,Db_B.shape[0]):
        date = Db_B.iloc[row,0]
        Btdy = Db_B.iloc[row,1]
        new_row = {'Date':date,'Btdy':Btdy}
        Db = Db._append(pd.Series(new_row,name=row))
    i = 0
    for row in range(0,Db_G.shape[0]):
        date = Db_G.iloc[row,0]
        Gtdy = Db_G.iloc[row,1]
        while Db.loc[i]['Date'] != date:
            i += 1
        Db.loc[i,'Gtdy'] = Gtdy
    return Db

if __name__ == '__main__':
    ##Db = Generation(Db_B,Db_G)
    ##print(Db)
    ##Db.to_csv('DATABASE.csv',index=False)
