##test

from heredity import *
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

##数据库初始条件
Db = pd.read_csv('DATABASE_1.csv')
Db.loc[4,'Call'] = 1000
Db.loc[4,'Ball'] = 0
Db.loc[4,'Gall'] = 0

class Invest:
    def __init__(self,dlr,bit,gld,bittdy,gldtdy,bitpdt,gldpdt):
        ##手续费用
        self.bitrat = 0.98
        self.gldrat = 0.99
        ##初始值
        self.dlr    = dlr
        self.bit    = bit
        self.gld    = gld
        self.bittdy = bittdy
        self.gldtdy = gldtdy
        self.bitpdt = bitpdt
        self.gldpdt = gldpdt

    def Trans(self,bittrs,gldtrs):
        self.bitrst = self.bit + bittrs
        self.gldrst = self.gld + gldtrs
        self.dlrrst = self.dlr - self.Check(bittrs,self.bitrat) - self.Check(gldtrs,self.gldrat)
    
    def restrict(self,bittrs,gldtrs):
        ##print("[",-self.Check(bittrs,self.bitrat) - self.Check(gldtrs,self.gldrat) + self.dlr,"]")
        if self.Check(bittrs,self.bitrat) + self.Check(gldtrs,self.gldrat) > self.dlr:
            ##print("out")
            return False
        else:
            return True

    def Check(self,subj,rate):
        try:
            temp = subj
            for i in range(0,subj.shape[0]):
                if subj[i] < 0:
                    temp[i] = subj[i] * rate
                else:
                    temp[i] = subj[i] / rate
            return temp
        except:
            if subj < 0:
                return subj * rate
            else:
                return subj / rate
    
    def demoFun(self,bittrs,gldtrs):
        self.Trans(bittrs,gldtrs)
        return self.dlrrst + self.bitrst * self.bitpdt / self.bittdy + self.gldrst * self.gldpdt / self.gldtdy

    def Train(self):
        if self.gldtdy != 1:
            self.train = Heredity(bound_x_arr=[-self.bit-1000,50000],       
                                  bound_y_arr=[-self.gld-1000,50000],
                                  function=self.demoFun,   isMax=True,
                                  restrict=self.restrict)
        else:
            self.train = Heredity(bound_x_arr=[-self.bit,40000],       
                                  bound_y_arr=[0,0.001],
                                  function=self.demoFun,   isMax=True,
                                  restrict=self.restrict)
        x, y = self.train.generate()
        self.bitrst = self.bit + x
        self.gldrst = self.gld + y
        if self.bitrst < 0: self.bitrst = 0
        if self.gldrst < 0: self.gldrst = 0
        self.Fix()
        ##self.Trans(x, y)
    
    def Fix(self):
        '''if 0.78 < self.demoFun(self.bitrst - self.bit, self.gldrst - self.gld) / (self.bit + self.gld + self.dlr) < 1:
            if self.bitpdt / self.bittdy < 0.78:
                self.bitrst = 0
            else:
                self.bitrst = self.bit
            if self.gldpdt / self.gldtdy < 0.78:
                self.gldrst = self.gld
            else:
                self.dlrrst = self.dlr'''
        self.Trans(self.bitrst - self.bit, self.gldrst - self.gld)
        print(self.bitrst, self.gldrst, self.dlrrst, self.demoFun(self.bitrst - self.bit, self.gldrst - self.gld))

if __name__ == '__main__':
    for i in range(4,Db.shape[0]-1):
        print("交易",i,"/",Db.shape[0]-1)
        if Db.loc[i,'Gtdy'] == Db.loc[i,'Gtdy']:
            bitpdt = Db.loc[i,'Bpdt']
            bittdy = Db.loc[i,'Btdy']
            gldpdt = Db.loc[i,'Gpdt']
            gldtdy = Db.loc[i,'Gtdy']
            bit = Db.loc[i,'Ball']
            gld = Db.loc[i,'Gall']
            dlr =  Db.loc[i,'Call']
            Train = Invest(dlr,bit,gld,bittdy,gldtdy,bitpdt,gldpdt)
            Train.Train()
            Db.loc[i,'Brst'] = Train.bitrst
            Db.loc[i,'Grst'] = Train.gldrst
            Db.loc[i,'Crst'] = Train.dlrrst
            Db.loc[i+1,'Ball'] = Train.bitrst * Db.loc[i+1,'Bpdt'] / bittdy
            Db.loc[i+1,'Call'] = Train.dlrrst
            while Db.loc[i+1,'Gtdy'] != Db.loc[i+1,'Gtdy']:
                i += 1
            Db.loc[i+1,'Gall'] = Train.gldrst * Db.loc[i+1,'Gpdt'] / gldtdy
        else:
            bitpdt = Db.loc[i,'Bpdt']
            bittdy = Db.loc[i,'Btdy']
            gldpdt = 0
            gldtdy = 1
            bit = Db.loc[i,'Ball']
            gld = 0
            dlr =  Db.loc[i,'Call']
            Train = Invest(dlr,bit,gld,bittdy,gldtdy,bitpdt,gldpdt)
            Train.Train()
            Db.loc[i,'Brst'] = Train.bitrst
            Db.loc[i,'Crst'] = Train.dlrrst
            Db.loc[i+1,'Ball'] = Train.bitrst * Db.loc[i+1,'Bpdt'] / bittdy
            Db.loc[i+1,'Call'] = Train.dlrrst

        Db.to_excel('DATABASE##__.xlsx',index=False)
        continue