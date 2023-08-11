import scipy
import numpy as np
from math import sqrt
from scipy import stats
from numpy import cov, linspace
from statistics import mean
from numpy.random import normal, exponential
import matplotlib.pyplot as plt
from random import random
%matplotlib inline


def lift(A, B):
    return mean(B) - mean(A)

def p_value(A, B):
    return stats.ttest_ind(A, B)[1]

def get_AB_samples(mu, sigma, treatment_lift, N):
    A = list(normal(loc=mu                 , scale=sigma, size=N))
    B = list(normal(loc=mu + treatment_lift, scale=sigma, size=N))
    return A, B

N = 1000
N_multiplier = 4
mu = 100
sigma = 10
treatment_lift = 2
num_simulations = 1000

print('Simulating %s A/B tests, true treatment lift is %d...' % (num_simulations, treatment_lift))

n1_lifts, n4_lifts = [], []
for i in range(num_simulations):
    print('%d/%d' % (i, num_simulations), end='\r')
    A, B = get_AB_samples(mu, sigma, treatment_lift, N)
    n1_lifts.append(lift(A, B))
    A, B = get_AB_samples(mu, sigma, treatment_lift, N_multiplier*N)
    n4_lifts.append(lift(A, B))

print('N samples  A/B testing, mean lift = %.2f, variance of lift = %.2f' % (mean(n1_lifts), cov(n1_lifts)))
print('4N samples A/B testing, mean lift = %.2f, variance of lift = %.2f' % (mean(n4_lifts), cov(n4_lifts)))
print('Raio of lift variance = %.2f (expected = %.2f)' % (cov(n4_lifts)/cov(n1_lifts), 1/N_multiplier))

bins = linspace(-2, 6, 100)
plt.figure(figsize=(14, 7))
plt.hist(n1_lifts, bins, alpha=0.5, label='N samples')
plt.hist(n4_lifts, bins, alpha=0.5, label=f'{N_multiplier}N samples')
plt.xlabel('lift')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.title('lift histogram')
plt.show()
# =============================================================================
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import product
from scipy.stats import norm, binom, mannwhitneyu
from tqdm import tqdm

# Сгенерированные данные. В вашем случае здесь нужно загрузить данные за продолжительный
# исторический период. Это необходимо, чтобы учесть поведение клиента и распределение 
# оригинальной метрики
np.random.seed(1)
mu = 42.
sd = 10.
data = norm.rvs(loc=mu, scale=sd, size=10000)
alpha = 0.05 # 1 - уровень значимости
simulations = 1000 # количество симуляций
lifts = np.arange(1, 1.1, 0.01) # последовательность шагов по эффекту

# Последовательность шагов по увеличению выборки. Я бы не рекомендовал использовать в живом проекте такой метод,
# т.к. не наследуется информация об окне метрики. Чтобы его учесть, лучше брать даты в качестве шага
sizes = np.arange(1000, 10001, 1000)

sim_res = pd.DataFrame() # сюда кладем результат расчетов

for lift, n in product(lifts, sizes): 
    print(lift, n)
    
    control = data[0:n]

    # В этом примере равномерно распределяем эффект по всему распределению
    test = control * lift 
    
    for _ in tqdm(range(0, simulations)):
        
        # Рандомное присвоение групп A/B
        is_control = binom.rvs(1, 0.5, size=n) 
        
        # Считаем p-value
        _, p = mannwhitneyu(control[is_control == True], test[is_control == False]) 

        # Кладем результат
        sim_res = sim_res.append({"lift": lift, "n": n, "pvalue": p}, ignore_index=True)
        
def calculate_tpr(df, sim_num):
    names = {
        "tpr": sum(df['pvalue'] < 0.05) / sim_num
    }
    return pd.Series(names)
    
res = sim_res.groupby(["lift", "n"]).apply(calculate_tpr, sim_num=simulations).reset_index()

display(res)
sns.lineplot(data=res, x="n", y="tpr", hue="lift")
# =============================================================================
# =============================================================================
import pandas as pd
import numpy as np

exp_list = []
exp_len = 1000
metric_list = [f"""metric_{i}""" for i in range(0,10)]

for id in range(0, exp_len):

    df = pd.DataFrame({
        "exp_id": id,
        "metric": metric_list,
        "lift": np.random.uniform(-0.1, 0.1, len(metric_list)),
        "statistic": np.random.normal(0, 1, len(metric_list))
    })

    exp_list.append(df)

exp_res = pd.concat(exp_list)
exp_res

ef agreement(row_i, row_j, tq = 1.96):
    """Проверка направленности"""
    cond_list = [
        (row_i.lift > 0) & (row_j.lift > 0) & (abs(row_i.statistic) < tq) & (abs(row_j.statistic) < tq),
        (row_i.lift < 0) & (row_j.lift < 0) & (abs(row_i.statistic) < tq) & (abs(row_j.statistic) < tq),
        (row_i.lift == 0) & (row_j.lift == 0) & (abs(row_i.statistic) < tq) & (abs(row_j.statistic) < tq)
    ]
    choice_list = [1,1,1]
    return np.select(cond_list,  choice_list, default=0)


agreements = np.zeros((10, 10))

for id in range(0, exp_len):
    df = exp_res[exp_res.exp_id == id]
    matrix = np.zeros((10, 10))

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            matrix[i][j] = agreement(row_i, row_j)

    agreements = agreements + matrix

agreements = np.triu(agreements, k=0)

exp_res = exp_res.pivot_table(index=['exp_id'],
         columns=["metric"], values='statistic') \
.reset_index()
exp_res

import matplotlib.pyplot as plt
import seaborn as sns

def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

sns.set(style='white', font_scale=1.6)
g = sns.PairGrid(exp_res, aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)
# =============================================================================
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import binom, ttest_ind
from tqdm import tqdm

# Сгенерированные данные. Представим, что моделируем экономику A/B в подписочном SaaS
pricing = np.array([0.99, 4.99, 9.99]) # цены в подписочной модели продукта

# Доля покупок по каждой цене. Представим гипотезу, в которой мы хотим увеличить долю
# платящих клиентов по самой дорогой цене, немного снизив долю по низкой
proportions_control = np.array([0.5, 0.4, 0.1]) 
proportions_test = np.array([0.49, 0.4, 0.11])

N = 10000
sizes = np.arange(1000, 10001, 1000)
simulations = 1000
sim_res = pd.DataFrame() 

np.random.seed(1)
control_pop = np.random.choice(pricing, N, p=proportions_control)
test_pop = np.random.choice(pricing, N, p=proportions_test)

for n in sizes: 
    
    control = control_pop[0:n]
    test = test_pop[0:n]
    
    for _ in range(0, simulations):
        
        # Рандомное присвоение групп A/B
        is_control = binom.rvs(1, 0.5, size=n)
        
        # Считаем p-value
        _, p = ttest_ind(control[is_control == True], test[is_control == False]) 

        # Кладем результат
        sim_res = sim_res.append({"n": n, "pvalue": p}, ignore_index=True)
        
def calculate_tpr(df, sim_num):
    names = {
        "tpr": sum(df['pvalue'] < 0.05) / sim_num
    }
    return pd.Series(names)
    
res = sim_res.groupby(["n"]).apply(calculate_tpr, sim_num=simulations).reset_index()

display(res)
sns.lineplot(data=res, x="n", y="tpr")
# =============================================================================
# =============================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind
from tqdm import tqdm
from matplotlib import pyplot as plt

def deltamethod(x_0, y_0, x_1, y_1):
    n_0 = y_0.shape[0]
    n_1 = y_0.shape[0]

    mean_x_0, var_x_0 = np.mean(x_0), np.var(x_0)
    mean_x_1, var_x_1 = np.mean(x_1), np.var(x_1)

    mean_y_0, var_y_0 = np.mean(y_0), np.var(y_0)
    mean_y_1, var_y_1 = np.mean(y_1), np.var(y_1)

    cov_0 = np.mean((x_0 - mean_x_0.reshape(-1, )) * (y_0 - mean_y_0.reshape(-1, )))
    cov_1 = np.mean((x_1 - mean_x_1.reshape(-1, )) * (y_1 - mean_y_1.reshape(-1, )))

    var_0 = var_x_0 / mean_y_0 ** 2 + var_y_0 * mean_x_0 ** 2 / mean_y_0 ** 4 - 2 * mean_x_0 / mean_y_0 ** 3 * cov_0
    var_1 = var_x_1 / mean_y_1 ** 2 + var_y_1 * mean_x_1 ** 2 / mean_y_1 ** 4 - 2 * mean_x_1 / mean_y_1 ** 3 * cov_1

    rto_0 = np.sum(x_0) / np.sum(y_0)
    rto_1 = np.sum(x_1) / np.sum(y_1)

    statistic = (rto_1 - rto_0) / np.sqrt(var_0 / n_0 + var_1 / n_1)
    pvalue = 2 * np.minimum(norm(0, 1).cdf(statistic), 1 - norm(0, 1).cdf(statistic))
    return pvalue

np.random.seed(3)
n = 1000

df = pd.DataFrame({
    'user_cnt': 1,
    'revenue_amt': np.random.exponential(100, size=n),
    'variant': np.random.randint(low = 0, high = 2, size = n)
})

display(df)

deltamethod(
    df.revenue_amt[df.variant == 0],
    df.user_cnt[df.variant == 0],
    df.revenue_amt[df.variant == 1],
    df.user_cnt[df.variant == 1]
)

ttest_ind(df.revenue_amt[df.variant == 0], df.revenue_amt[df.variant == 1])

n = 1000

pvalues_dm = []
pvalues_t = []
np.random.seed(4)
for _ in tqdm(range(0,1000)):
    sim = pd.DataFrame({
        'user_cnt': 1,
        'revenue_amt': np.random.exponential(100, size=n),
        'variant': np.random.randint(low = 0, high = 2, size = n)
    })
    sim['variant'] = np.random.randint(0,2,len(sim))

    pvalues_dm.append(deltamethod(
        sim.revenue_amt[sim.variant == 0],
        sim.user_cnt[sim.variant == 0],
        sim.revenue_amt[sim.variant == 1],
        sim.user_cnt[sim.variant == 1]
    ))
    pvalues_t.append(ttest_ind(sim.revenue_amt[sim.variant == 0], sim.revenue_amt[sim.variant == 1])[1])

print(f"""
t-test Deltamethod: {float(sum(np.array(pvalues_dm) <= 1 - 0.95) / len(pvalues_dm))},
t-test: {float(sum(np.array(pvalues_t) <= 1 - 0.95) / len(pvalues_t))}
""")

# Сравненим как далеки друг от друга pvalue двух критериев
f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pvalues_t, pvalues_dm, c=".1")
ax.plot([0, 1], [0, 1], ls="--", c=".1")
