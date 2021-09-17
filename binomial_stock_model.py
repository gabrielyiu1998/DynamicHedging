import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime
import pandas as pd
import scipy.stats as sc
  
def growtree(s_0, depth, up, down):
    """
    Grows binary tree information

    Parameters
    ----------
    s_0 : float
        The initial stock price.
    depth : int
        The depth of the tree (i.e. the number of periods to be considered).
    up : float
        The multiple increase per period.
    down : float
        The multiple decrease per period.

    Returns
    -------
    dict.

    """
    tree = {}
    for i in range(depth + 1):
        for j in range(depth + 1 - i):
            tree[(i, j)] = s_0 * (up ** i) * (down ** j)
    return tree
 
def periods(t):
    """
    Gets number of periods in tree.

    Parameters
    ----------
    t : dict
        An output of growtree.

    Returns
    -------
    int.

    """
    return max([sum(k) for k in t])
    
def derivative(stock_path, h, r, der):
    """
    Evaluates the value of the derivative der with risk free rate r and
    timestep h.

    Parameters
    ----------
    stock_path : BinaryTree
        The stock path to evaluate the option on.
    h : float
        The length of each step.
    r : float
        Risk free rate.
    der : function
        A function that evaluates the payoff given the stock price.

    Returns
    -------
    BinaryTree.
    
    """    
    tree = {}
    pds = periods(stock_path)
    for i in range(pds + 1):
        tree[(i, pds - i)] = der(stock_path[(i, pds - i)])
    if pds > 0:
        up = stock_path[(1, 0)] / stock_path[(0, 0)]
        down = stock_path[(0, 1)] / stock_path[(0, 0)]
        p = (np.exp(r * h) - down) / (up - down)
        for i in reversed(range(pds)):
            for j in range(i + 1):
                tree[(j, i - j)] = (np.exp(-r * h) * 
                                    (p * tree[(j + 1, i-j)] +
                                     (1 - p) * tree[(j, i - j + 1)]))
    return tree

def delta_hedge(stock_path, h, r, der, path = (0, 0)):
    """
    Evaluates the delta hedge of the derivative der with risk free rate r and
    timestep h.

    Parameters
    ----------
    stock_path : BinaryTree
        The stock path to evaluate the option on.
    h : float
        The length of each step.
    r : float
        Risk free rate.
    der : function
        A function that evaluates the payoff given the stock price.

    Returns
    -------
    np.array
    [Delta, Cash]
    
    """
    up = (path[0] + 1, path[1])
    down = (path[0], path[1] + 1)
    if stock_path.keys() == {(0, 0)}:
        return np.array([0,0])
    if not (path[0] + 1, path[1]) in stock_path.keys():
        raise Exception("Either termination path or path does not exist.")
    der_tree = derivative(stock_path, h, r, der)
    mat = np.array([[stock_path[up], np.exp(r * h)],
                  [stock_path[down], np.exp(r * h)]])
    payoffs = np.array([der_tree[up], der_tree[down]])
    return np.linalg.solve(mat, payoffs)

def roundtree(binom_tree, dec = 2):
    """
    Prints rounded values.

    Parameters
    ----------
    t : BinaryTree
        
    Returns
    -------
    None.

    """
    tree = {}
    for key in binom_tree:
        tree[key] = round(binom_tree[key], dec)
    return tree

def visualize(binom_tree, tot_time, y_values = False, annotate = True):
    """
    Plots the binomial tree on time - values plane.

    Parameters
    ----------
    binom_tree : dict
        A binomial tree of values.
    y_values : bool, optional
        Plot values on y axis
    annotate : bool, optional
        Annotates y values
    Returns
    -------
    None.

    """
    a = 0.2   
    n = periods(binom_tree)
    if y_values:
        for t in range(n):
            for y in range(t + 1):
                times = np.array([t / n * tot_time, (t + 1) / n * tot_time])
                values = np.array([binom_tree[(y, t - y)],
                                   binom_tree[(y, t - y + 1)]])
                plt.plot(times, values,'bo-', alpha = a, markersize =1)
                if annotate and t == n - 1:
                    plt.text(times[1] + 0.01, values[1] + 0.01,
                             round(values[1], 2))
                times = np.array([t / n * tot_time, (t + 1) / n * tot_time])
                values = np.array([binom_tree[(y, t - y)],
                                   binom_tree[(y + 1, t - y)]])
                plt.plot(times, values,'bo-', alpha = a, markersize =1)
                if annotate and t == n - 1:
                    plt.text(times[1] + 0.01, values[1] + 0.01,
                             round(values[1],2))
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
        plt.show()
    else:
        t = 0
        ys = [0]
        if annotate:
            plt.text(0.01, 0.01,round(binom_tree[(0, 0)],2))
        for t in range(n):
            new = []
            new.append(ys[0] + 1)
            if annotate:
                plt.text((t + 1) / n * tot_time + 0.01, ys[0] + 1.01,
                         round(binom_tree[(t + 1, 0)], 2))
            for i in range(t + 1):
                times = np.array([t / n * tot_time, (t + 1) / n * tot_time])
                values = np.array([ys[i], ys[i] + 1])
                plt.plot(times, values,'bo-', alpha = a)
                times = np.array([t / n * tot_time, (t + 1) / n * tot_time])
                values = np.array([ys[i], ys[i] - 1])
                new.append(ys[i] - 1)
                if annotate:
                    plt.text(times[1] + 0.01, values[1] + 0.01,
                             round(binom_tree[(t - i, i + 1)],2))
                plt.plot(times, values,'bo-', alpha = a)
            ys = new
        plt.axis('off')
        plt.show()

def visual_path(steps):
    """
    Visualizes tree with latex text.

    Parameters
    ----------
    steps : int
        Number of periods

    Returns
    -------
    None.

    """
    t = 0
    ys = [0]
    plt.text(0.01, 0.01,r'$S_0$')
    while t < steps:
        new = []
        new.append(ys[0] + 1)
        if t == 0:
            txt = r'$S_0 u$'
        else:
            txt = r'$S_0 u^{}$'.format(t + 1)
        plt.text(t + 1.01, ys[0] + 1.01, txt)
        for i in range(t + 1):
            times = np.array([t, (t + 1)])
            values = np.array([ys[i], ys[i] + 1])
            plt.plot(times, values,'bo-', alpha = 0.2)
            times = np.array([t, t + 1])
            values = np.array([ys[i], ys[i] - 1])
            new.append(ys[i] - 1)
            if i == 0:
                ds = 'd'
            else:
                ds = 'd^{}'.format(i + 1)
            if i == t - 1:
                us = 'u'
            elif i == t:
                us = ''
            else:
                us = 'u^{}'.format(t - i)
            txt = r'$S_0' + us + ds +'$'
            plt.text(times[1] + 0.01, values[1] + 0.01, txt)
            plt.plot(times, values,'bo-', alpha = 0.2)
        ys = new
        t += 1
    plt.axis('off')
    plt.show()

def backtest(S_0, r, g, tot_time, index,
             start = '2019-03-30', finish = '2020-03-30'):
    
    stock_data = pdr.DataReader(index, 'yahoo', start, finish)
    stock_data = stock_data.asfreq('W-TUE')['Adj Close']
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    
    sig = 0.15 #Standard estimate of volatility fo S&P500
    n = len(returns)
    h = 1 / n
    put = lambda x: max(S_0 * np.exp(g * tot_time) - x, 0)
    t_tree = growtree(S_0, n,
                      np.exp(r * h + sig * np.sqrt(h)),
                      np.exp(r * h - sig * np.sqrt(h)))
    p_tree = derivative(t_tree, h, r, put)
    hedge_values = delta_hedge(t_tree, h, r, put)
    hedge_costs = [p_tree[(0,0)]]
    profit_loss = [0]
    value_of_index = [S_0]
    units_held = [hedge_values[0]]
    cash_position = [hedge_values[1]]
    for time in range(n):
        value_of_index.append(value_of_index[time] * np.exp(returns[time]))
        t_tree = growtree(value_of_index[time] *
                          np.exp(returns[time]), n - time - 1,
                          np.exp(r * h + sig * np.sqrt(h)),
                          np.exp(r * h - sig * np.sqrt(h)))
        p_tree = derivative(t_tree, h, r, put)
        hedge_values = delta_hedge(t_tree, h, r, put)
        units_held.append(hedge_values[0])
        cash_position.append(hedge_values[1])
        hedge_costs.append(p_tree[(0,0)])
        profit_loss.append(units_held[time] * t_tree[(0,0)] +
                           cash_position[time] * np.exp(r * h) - p_tree[(0,0)])
        
    d = {'date': stock_data.index,
         'hedge_costs': hedge_costs, 'profit_loss': profit_loss,
         'value_of_index': value_of_index, 'units_held': units_held,
         'cash_position': cash_position}
    
    data_f = pd.DataFrame(d)
    data_f = data_f.set_index('date')
    return data_f    

if __name__ == '__main__':    
    r = 0.04
    g = 0.02
    S_0 = 100
    h = 1
    df = backtest(S_0, r, g, 1, '^GSPC')
    plt.figure(figsize=(9, 6), dpi=100)
    plt.rc('font', size = 14)
    plt.plot(df['hedge_costs'])
    plt.title('Hedge Costs')
    plt.show()
    plt.figure(figsize=(9, 6), dpi=100)
    plt.rc('font', size = 14)
    plt.plot(pd.Series.cumsum(df['profit_loss']))
    plt.title('Cumulative Profit or Loss')
    plt.show()
    print("Cumulative profit or loss: {}".format(
        round(sum(df['profit_loss']), 2)))
    
    # Final Slide
    plt.figure(figsize=(16, 9), dpi=100)
    plt.rc('font', size = 18)
    plt.ylim([55, 190])
    plt.plot(df['value_of_index'], color = 'red', alpha = 0.7)
    plt.xlabel('Time')
    plt.ylabel('Value ($)')
    
    plt.figure(figsize=(16, 9), dpi = 100)
    plt.rc('font', size = 18)
    plt.ylim([55, 190])
    visualize(growtree(100, 51,
                       np.exp(0.04 / 51 + 0.08 * np.sqrt(1/51)),
                       np.exp(0.04 / 51 - 0.08 * np.sqrt(1/51))),
              1, True, False)
    
    cx_calcs = False
    if cx_calcs:    
        print("Example 1")
        r = 0.04
        sig = 0.08
        g = 0.02
        S_0 = 100
        h = 1
        ex1 = growtree(S_0, 1, np.exp(r + sig), np.exp(r - sig))
        put = lambda x: max(S_0 * np.exp(g) - x, 0)
        p_tree1 = derivative(ex1, h, r, put)
        hedge_values = delta_hedge(ex1, h, r, put)
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of risk-free bond".format(
            round(hedge_values[1], 2)))
        print("{} is the hedge cost.".format(round(p_tree1[(0, 0)], 2)))
        visual_path(1)
        visualize(ex1, 1, True)
        visualize(p_tree1, 1)
        
        print("\n Example 2")
        r = 0.04
        sig = 0.08
        g = 0.02
        S_0 = 100
        h = 1/2
        ex2 = growtree(S_0, int(1 / h),
                       np.exp(r * h + sig * np.sqrt(h)),
                       np.exp(r * h - sig * np.sqrt(h)))
        put = lambda x: max(S_0 * np.exp(g) - x, 0)
        p_tree2 = derivative(ex2, h, r, put)
        hedge_values = delta_hedge(ex2, h, r, put)
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of risk-free bond".format(
            round(hedge_values[1], 2)))
        print("{} is the hedge cost.".format(round(p_tree2[(0, 0)], 2)))
        visual_path(2)
        visualize(ex2, 1, True)
        visualize(p_tree2, 1)
        
        print("\n Hedge of up tree.")
        hedge_values = delta_hedge(ex2, h, r, put, path = (1, 0))
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of stock risk-free bond".format(
            round(hedge_values[1], 2)))
        
        print("\n Hedge of down tree.")
        hedge_values = delta_hedge(ex2, h, r, put, path = (0, 1))
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of risk-free bond".format(
            round(hedge_values[1], 2)))
        print("{} is the hedge cost.".format(round(p_tree2[(0, 1)], 2)))
        
        print("\n Example 3")
        plt.rc('font', size = 8)
        r = 0.04
        sig = 0.08
        g = 0.02
        S_0 = 100
        h = 1/3
        ex3 = growtree(S_0, int(1 / h),
                       np.exp(r * h + sig * np.sqrt(h)),
                       np.exp(r * h - sig * np.sqrt(h)))
        put = lambda x: max(S_0 * np.exp(g) - x, 0)
        p_tree3 = derivative(ex3, h, r, put)
        hedge_values = delta_hedge(ex3, h, r, put)
        print("{} units of stock market portfolio".format(
            round(hedge_values[0], 4)))
        print("{} units of risk-free bond".format(
            round(hedge_values[1], 2)))
        print("{} is the hedge cost.".format(round(p_tree3[(0, 0)], 2)))
        visual_path(3)
        visualize(ex3, 1, True)
        visualize(p_tree3, 1)
    

            
            
            
            
            
            