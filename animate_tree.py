import binomial_stock_model as bsm
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import pandas as pd
import scipy.stats as sc

if __name__ == '__main__':
    stock_data = pdr.DataReader('^GSPC', 'yahoo', '2019-03-30', '2020-03-30')
    stock_data = stock_data.asfreq('W-TUE')['Adj Close']
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    
    r = 0.04
    sig = 0.08
    g = 0.02
    S_0 = 100
    h = 1/2
    ex4 = bsm.growtree(S_0, int(1 / h),
                       np.exp(r * h + sig * np.sqrt(h)),
                       np.exp(r * h - sig * np.sqrt(h)))
    put = lambda x: max(S_0 * np.exp(g) - x, 0)
    totalT = 1    
    annotate = True
    
    anim = False
    
    if anim:
        annotate = True
        plt.figure(figsize = (10, 8), dpi = 500)
        plt.rc('font', size = 18)
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
           
        n = bsm.periods(ex4)
        if annotate:
            plt.text(0, ex4[(0,0)],round(ex4[(0, 0)],2))
        for t in range(n):
            for y in range(t + 1):
                times = np.array([t / n, (t + 1) / n])
                values = np.array([ex4[(y, t - y)],
                                   ex4[(y, t - y + 1)]])
                plt.plot(times, values,'bo-', alpha = 0.2)
                if annotate:
                    plt.rc('font', size = 8)
                    plt.text(times[1] + 0.01, values[1] + 0.01,
                             round(values[1], 2))
                times = np.array([t / n, (t + 1) / n])
                values = np.array([ex4[(y, t - y)],
                                   ex4[(y + 1, t - y)]])
                plt.plot(times, values,'bo-', alpha = 0.2)
                if annotate:
                    plt.rc('font', size = 8)
                    plt.text(times[1] + 0.01, values[1] + 0.01,
                             round(values[1],2))
        plt.draw()
        ex5 = bsm.growtree(105, 1,
                           np.exp(r * h + sig * np.sqrt(h)),
                           np.exp(r * h - sig * np.sqrt(h)))
        plt.plot([0.5], [ex5[(0,0)]], 'ro', alpha = 0.2)
        plt.text(0.51, 105.01, "105")
        plt.draw()
        plt.plot([0.5, 1], [105, ex5[(1, 0)]],
                  'ro-', alpha = 0.2)
        plt.text(1.01, ex5[(1, 0)] + 0.01,
                  round(ex5[(1, 0)], 2))
        plt.plot([0.5, 1], [105, ex5[(0, 1)]],
                  'ro-', alpha = 0.2)
        plt.text(1.01, ex5[(0, 1)] + 0.01,
                  round(ex5[(0, 1)], 2))
        plt.draw()
        
        p_tree5 = bsm.derivative(ex5, h, r, put)
        hedge_values = bsm.delta_hedge(ex5, h, r, put)
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of risk-free bond".format(
            round(hedge_values[1], 2)))
        print("{} is the hedge cost.".format(round(p_tree5[(0, 0)], 2)))
        print("{} position from t=0".format(
            round(bsm.delta_hedge(ex4, h, r, put)[0] * 105 +
                  bsm.delta_hedge(ex4, h, r, put)[1] * np.exp(r * h))))
        
    r = 0.04
    sig = 0.15
    g = 0.02
    S_0 = 100
    h = 1/51
    ex4 = bsm.growtree(S_0, int(1 / h),
                       np.exp(r * h + sig * np.sqrt(h)),
                       np.exp(r * h - sig * np.sqrt(h)))
    put = lambda x: max(S_0 * np.exp(g) - x, 0)
    totalT = 1    
    annotate = True
    n = int(1/h)
    
    anim = False
    
    if anim:
        annotate = True
        
        df = pd.DataFrame([[S_0]], columns = ['S&P500 Movement'],
                  index = [stock_data.index[0]])
        for t in range(n):
            plt.figure(figsize=(16, 9), dpi=100)
            plt.rc('font', size = 18)
            plt.xlabel('Time')
            plt.ylabel('Value ($)')
            plt.ylim([55, 190])
            tempt = bsm.growtree(df['S&P500 Movement'][t], n - t,
                                 np.exp(r * h + sig * np.sqrt(h)),
                                 np.exp(r * h - sig * np.sqrt(h)))
            for tx in range(n - t):
                for y in range(tx + 1):
                    times = np.array([stock_data.index[tx + t], stock_data.index[tx + t + 1]])
                    values = np.array([tempt[(y, tx - y)],
                                        tempt[(y, tx - y + 1)]])
                    if annotate and tx == n - t - 1:
                        plt.rc('font', size = 8)
                        plt.text(times[1] + pd.DateOffset(1), values[1] + 0.01,
                                  round(values[1], 2))
                    plt.plot(pd.DataFrame(values, columns = ['Tree'], index = times),
                              'ro-', alpha = 0.2, markersize = 1)
                    times = np.array([stock_data.index[tx + t], stock_data.index[tx + t + 1]])
                    values = np.array([tempt[(y, tx - y)],
                                        tempt[(y + 1, tx - y)]])
                    plt.plot(pd.DataFrame(values, columns = ['Tree'], index = times),
                              'ro-', alpha = 0.2, markersize = 1)
                    if annotate and tx == n - t - 1:
                        plt.rc('font', size = 8)
                        plt.text(times[1] + pd.DateOffset(1), values[1] + 0.01,
                                  round(values[1],2))
            plt.plot(df, color = 'red', alpha = 0.7)
            plt.draw()
            plt.pause(0.1)
            df_temp = pd.DataFrame([[df['S&P500 Movement'][stock_data.index[t]] * np.exp(returns[t])]],
                                   columns = ['S&P500 Movement'],
                                   index = [stock_data.index[t + 1]])
            df = df.append(df_temp)
        plt.show()
        plt.rc('font', size = 14)
    
    anim = False
    
    if anim:
        t = 0
        plt.figure(figsize=(16, 9), dpi=100)
        plt.draw()
        plt.pause(5)
        for k in range(n):
            plt.yticks([])
            ys = [0]
            h = 1/(k + 1)
            if annotate:
                temptree = bsm.growtree(S_0, k + 1,
                                        np.exp(r * h + sig * np.sqrt(h)),
                                        np.exp(r * h - sig * np.sqrt(h)))
                ptree = bsm.derivative(temptree, h, r, put)
                # print(round(ptree[(0,0)],2))
                plt.text(0.01, 0.01, round(ptree[(0, 0)], 2))
            for t in range(k):
                new = []
                new.append(ys[0] + 1)
                for i in range(t + 1):
                    times = np.array([t / n * totalT, (t + 1) / n * totalT])
                    values = np.array([ys[i], ys[i] + 1])
                    plt.plot(times, values,'bo-', alpha = 0.2, markersize = 1)
                    times = np.array([t / n * totalT, (t + 1) / n * totalT])
                    values = np.array([ys[i], ys[i] - 1])
                    new.append(ys[i] - 1)
                    plt.plot(times, values,'bo-', alpha = 0.2, markersize = 1)
                ys = new
            plt.draw()
            plt.pause(0.1)
            plt.clf()
        plt.show()
        