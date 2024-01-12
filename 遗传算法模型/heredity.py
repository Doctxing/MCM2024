##遗传算法模型,用于求二元变量

import numpy as np

class Heredity:
    def __init__(self,bound_x_arr,bound_y_arr,function,isMax,restrict):
        ##种群·神職设置(默认值)
        self.DNA_SIZE = 24  # 编码长度，不宜太大
        self.POP_SIZE = 1000  # 种群大小
        self.CROSS_RATE = 0.5  # 交叉率
        self.MUTA_RATE = 0.15  # 变异率
        self.Iterations = 100  # 迭代次数
        ##变异区间
        self.X_BOUND = bound_x_arr
        self.Y_BOUND = bound_y_arr
        ##对象函数
        self.function = function
        ##求二元函数的最大值或是最小值
        self.isMax = isMax
        ##限制表达式
        self.restrict = restrict
    
    def getfitness(self,pop):  # 适应度函数
        x, y = self.decodeDNA(pop)
        temp = self.function(x, y)
        if self.isMax == True:
            return (temp-np.min(temp))+0.0001 ##0.0001防止误判为负数
        else:
            return (np.max(temp)-temp)+0.0001
    
    def decodeDNA(self,pop):  # 二进制转坐标，解码（说实话这部分代码还没弄懂，不过不影响使用）
        x_pop = pop[:, 1::2]
        y_pop = pop[:, ::2]
        # .dot()用于矩阵相乘
        x = x_pop.dot(2**np.arange(self.DNA_SIZE)[::-1])/float(2**self.DNA_SIZE-1)*(self.X_BOUND[1]-self.X_BOUND[0])+self.X_BOUND[0]
        y = y_pop.dot(2**np.arange(self.DNA_SIZE)[::-1])/float(2**self.DNA_SIZE-1)*(self.Y_BOUND[1]-self.Y_BOUND[0])+self.Y_BOUND[0]
        return x, y

    def select(self,pop,fitness):  # 使用随机抽样模拟自然选择
        if self.restrict != None:
            for i in range(0,self.POP_SIZE):
                x, y = self.decodeDNA(pop[[i]])
                if self.restrict(x, y) == False:
                    fitness[i] = 0
        temp = np.random.choice(np.arange(self.POP_SIZE),  ##样本空间为每条基因（pop）的下标
                                size=self.POP_SIZE,        ##抽取样本为种群数量
                                replace=True,              ##出现自然淘汰（放回式抽样）
                                p=fitness/(fitness.sum())) ##适应度（fitness）表现越好越不容易被淘汰
        return pop[temp]                                   ##返回所有被选中的基因组

    def mutation(self,temp):  # 变异
        if np.random.rand() < self.MUTA_RATE:
            mutate_point = np.random.randint(0, self.DNA_SIZE)
            temp[mutate_point] = temp[mutate_point] ^ 1   # ^为异或运算
        return temp
 
    def crossmuta(self,pop):  # 交叉
        new_pop = []
        for i in pop:
            temp = i
            if np.random.rand()<self.CROSS_RATE:
                j = pop[np.random.randint(self.POP_SIZE)]
                cpoints1 = np.random.randint(0, self.DNA_SIZE*2-1)
                cpoints2 = np.random.randint(cpoints1, self.DNA_SIZE*2)
                temp[cpoints1:cpoints2] = j[cpoints1:cpoints2]
                temp = self.mutation(temp)
            new_pop.append(temp)
        return new_pop
    
    ##主调用函数
    def generate(self):
        pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE*2))
        for i in range(self.Iterations):
            print("迭代进度：",i,"/",self.Iterations)
            pop = np.array(self.crossmuta(pop))
            fitness = self.getfitness(pop)
            pop = self.select(pop,fitness)
        fitness = self.getfitness(pop)
        if self.isMax == True:
            temp = [np.argmax(fitness)]
        else:
            temp = [np.argmin(fitness)]
        solution_x, solution_y = self.decodeDNA(pop[temp])
        return *solution_x, *solution_y

##应用示例
if __name__ == '__main__':
    def demoFun(x, y): ##二元函数表达式
        return 100-(x-3)*(x-3)-(y-4)*(y-4)
    def restrict(x, y):  ##限制表达式，返回值为布尔值
        if x<4:
            return False
        else:
            return True
    Train = Heredity(bound_x_arr=[0,10], bound_y_arr=[0,10],
                     function=demoFun,   isMax=True,
                     restrict=restrict)
    ##此代码可用于调节种群·神職设置(默认值)
    # Train.Iterations = 100
    solution_x, solution_y = Train.generate()
    print(solution_x, solution_y)