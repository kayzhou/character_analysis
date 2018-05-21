import numpy as np
import scipy.stats as stats 


def my_corr(x1, x2):
    r, p=stats.pearsonr(np.array(x1), np.array(x2)) 
    print('Pearson相关系数：', r, 'p-value：', p)

