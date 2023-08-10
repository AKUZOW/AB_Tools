# =============================================================================
# 
# =============================================================================
import math
import numpy as np
import statsmodels.stats.power as smp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# Критерий пропорций (нужен для кликов, конверсий). Ниже посчитаем прогноз, в котором мы хотели бы увидеть разницу p_x и p_y.
# Когда подойдет время сравнения экспериментального эффекта с прогнозом, нам таким же образом нужно взять все экспериментальные
# параметры и сравнить их. Например, прогнозный h с фактическим (экспериментальным) h
alpha = 0.05
power = 0.8
n = 450
p_x = 0.25
p_y = 0.2875

h = 2*math.asin(np.sqrt(p_x)) - 2*math.asin(np.sqrt(p_y))

# Сколько наблюдений необходимо для заданного эффекта, чтобы
# результаты двухвыборочного теста пропорций были с заданным уровнем значимости 
# и заданной мощностью критерий (1 - вероятность ошибки II-го рода)
smp.zt_ind_solve_power(effect_size = h, alpha = alpha, power = power, alternative='two-sided')

# Какая мощность критерия будет в выборке из n наблюдений, 
# при условии, что величина эффекта = h c уровнем значимости alpha
power = smp.zt_ind_solve_power(effect_size = h, nobs1 = n, alpha = alpha, alternative='two-sided')
power

# Какова величина эффекта будет при рассчете теста, при заранее заданных
# величинах альфа и величины мощности 
smp.zt_ind_solve_power(nobs1 = n, alpha = alpha, power = power)
    

# T-test solving
effects = []
sample_sizes = []

for i in tqdm(range(50,2000)):
    effects.append(smp.tt_ind_solve_power(nobs1 = i, alpha = alpha, power = power))
    sample_sizes.append(i)

viz = sns.lineplot(x=sample_sizes, y=effects)
viz.set_xlabel("Sample Size")
viz.set_ylabel("Effect Amount")

# 1.Считаем кол-во наблюдений
from statsmodels.stats.power import tt_ind_solve_power
mean = 2021.989 # рассчитанное среднее за 2-5 недель
se = 2425.011 # рассчитанное стандартное отклонение за тот же период
power = 0.9
alpha = 0.005
lift = 0.1 # хотим увидеть 10% изменение в метрику
effect_size = mean / se * lift 
print(effect_size)

# 2. Размер выборки Х2
print(tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, nobs1=None, ratio=1))
print(2*tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, nobs1=None, ratio=1))

# 3. Какой MDE если выборка ограничена
from statsmodels.stats.power import tt_ind_solve_power
nobs = 2000 # возьмем то другое число
power = 0.9
alpha = 0.005

# Результат – допустимый минимальный effect_size 
print(tt_ind_solve_power(alpha=alpha, power=power, nobs1=nobs, ratio=1))

# =============================================================================
# 
# =============================================================================
import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.power import tt_ind_solve_power
import matplotlib.pyplot as plt
import seaborn as sns


N = 2000
n = int(N/2)
mu = 10 
sigma = 2
lift = 1.05
pwr = 0.8 
alpha = 0.05
es = (mu*lift - mu)/sigma

res = pd.DataFrame()

# Объявим тестовые данные
pd_df = pd.DataFrame({
    "val": np.concatenate([np.random.normal(mu, sigma, n), np.random.normal(mu*lift, sigma, n)]),
    "grp": np.concatenate([np.repeat("c",n),np.repeat("t",n)])
})
pd_df = pd_df.sample(frac=1)

# Кумулятивно считаем MDE в ходе эксперимента
for i in range(10,len(pd_df)):
    t = pd_df.iloc[:i, :]
    
    df = len(t) - 2 # степени свободы
    
    sd_t = np.std(t['val'][t['grp'] == 't'])
    sd_c = np.std(t['val'][t['grp'] == 'c'])
    
    n_t = len(t[t['grp'] == 't'])
    n_c = len(t[t['grp'] == 'c'])
    
    y_c = np.mean(t['val'][t['grp'] == 'c'])
    
    S = np.sqrt((sd_t**2 / n_t) + (sd_c**2 / n_c)) # SE
    M = scipy.stats.t.ppf(pwr, df) - scipy.stats.t.ppf(alpha / 2, df)  # расчет квантилей t-распределения
    
    # Calculate the p-value dynamically using scipy.stats.ttest_ind
    t_stat, p_val = scipy.stats.ttest_ind(t['val'][t['grp'] == 't'], t['val'][t['grp'] == 'c'])
    
    
    to_insert = {
        "n": len(t.index),
        "mde": M * S, # ES
        "mde_perc": M * S / y_c, # ES as % of the control mean
        "p_value": p_val
    }
    
    res = res.append(to_insert, ignore_index=True)


plt.plot(res['mde'])
plt.ylabel('mde')
plt.axhline(y=es, color='black', linestyle='--', label=f'Effect Size = {es}')
plt.show()

plt.plot(res['p_value'])
plt.ylabel('p-value')
plt.axhline(y=alpha, color='black', linestyle='--', label=f'alpha = {alpha}')
plt.show()
# =============================================================================
# 
# =============================================================================