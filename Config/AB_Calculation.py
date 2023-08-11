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
def agg_totals(df):    
    """Ф-ция расчета агрегат"""
    
    names = {
        "gmv": df['driver_bill'].sum(),
        "cost": df['fee'].sum(),
        "orders_count": df['orders_count'].sum(),
        'n': df['driver_id'].nunique(),
        "orders_per_driver": df['orders_count'].sum() / df['driver_id'].nunique(),
        "gmv_per_driver": df['driver_bill'].sum() / df['driver_id'].nunique()
    }
    return pd.Series(names)

def pivot_df(df, col_names=['metric_name', 'value_1', 'value_2']):
    """Пивот таблицы для удобной работы c расчетом fixed horizon"""
    
    df = pd.melt(df, id_vars=['split']) \
        .pivot_table(index=['variable'], columns=['split'], values='value') \
        .reset_index()
    df.columns = col_names
    return df


# Препроцесс
split_level_agg = pivot_df(
    df.groupby('split').apply(agg_totals).reset_index()
)

# отдельно считаем группировку по пользователям, чтобы независимо посчитать дисперсию (т.к. в оригинале дубли)
driver_level_agg = pivot_df(
    df.groupby(['split','driver_id']).apply(agg_totals).reset_index(), ['metric_name', 'std_1', 'std_2']
)

exp_res = split_level_agg.merge(driver_level_agg)
exp_res['n_1'] = int(split_level_agg['value_1'][split_level_agg.metric_name == 'n'])
exp_res['n_2'] = int(split_level_agg['value_2'][split_level_agg.metric_name == 'n'])

alpha = 0.05
power = 0.8

# расчет t-критерия из готовых статистик
_, exp_res['pvalue'] = ttest_ind_from_stats(
    mean1 = exp_res['value_1'], 
    std1 = exp_res['std_1'], 
    nobs1 = exp_res['n_1'], 
    mean2 = exp_res['value_2'], 
    std2 = exp_res['std_2'], 
    nobs2 = exp_res['n_2'], 
    equal_var = False)

# считаем стандартизированный эффект сайз с которым потом будем сравнивать mde
exp_res['lift'] = (exp_res['value_2'] - exp_res['value_1']) / exp_res['value_1']
exp_res['effect_size'] = abs(exp_res['value_1'] / exp_res['std_1'] * exp_res['lift'])

# Считаем мде и необходимое количество наблюдений. 
exp_res['mde'] = [tt_ind_solve_power(nobs1=row[0], alpha=alpha, power=power) for row in zip(exp_res['n_1'])]
exp_res['n_need'] = [tt_ind_solve_power(row[0], alpha=alpha, power=power) for row in zip(exp_res['effect_size'])]

# Увидим ошибки, но не обращаем внимания на них. В первом случае нереально большой эффект, а во втором дисперсия равная единице
# также не обращаем внимание на метрики n, orders_count, gmv
exp_res = exp_res[~exp_res.metric_name.isin(['n','orders_count','gmv'])]
display(exp_res)
# =============================================================================
# =============================================================================
# Stratification
# =============================================================================
treatment_effect = 1
# Объявим дф с 3 стратами, где у всех отличается дисперсия и средние
def gen_data(treatment_effect = 0):

    stratum_1 = pd.DataFrame({"group": "stratum_1", "val": norm.rvs(size=12000, loc=15 + treatment_effect, scale=2)})
    stratum_2 = pd.DataFrame({"group": "stratum_2", "val": norm.rvs(size=6000, loc=20 + treatment_effect, scale=2.5)})
    stratum_3 = pd.DataFrame({"group": "stratum_3", "val": norm.rvs(size=2000, loc=30 + treatment_effect, scale=3)})

    return pd.concat([stratum_1, stratum_2, stratum_3])

df_control = gen_data()
df_control["variant"] = "Control"
df_control["indx"] = df_control.index

df_treatment = gen_data(treatment_effect)
df_treatment["variant"] = "Treatment"
df_treatment["indx"] = df_treatment.index

df_combined = pd.concat([df_control, df_treatment]).reset_index()

sns.histplot(data=df_combined, x="val", hue="variant", element="poly")
g = sns.FacetGrid(df_combined, col="variant", hue="group")
g.map_dataframe(sns.histplot, x="val")
g.add_legend()

# without strat
normal_te = pd.DataFrame({
    "effect_estimate": np.mean(df_treatment.val - df_control.val),
    "effect_estimate_se": np.sqrt(np.var(df_treatment.val) / len(df_treatment.val) + np.var(df_control.val) / len(df_control.val)),
    "n": len(df_treatment.val) + len(df_control.val)
}, index=[0])
print(normal_te)

def get_effect_estimate_se(treatment, control):
    return np.sqrt(np.var(treatment) / len(treatment) + np.var(control) / len(control))

def get_effect_estimate(treatment, control):
    return np.mean(treatment - control)

# with strats
groups = {}
for k, group in df_combined.groupby(by="group"):
    cur_df = pd.DataFrame()
    for g, variant in group.groupby(by="variant"):
        cur_df = pd.concat([cur_df, variant])
    groups[k] = cur_df

effect_estimate_se = []
effect_estimate = []
n = []
stratums = []

for key, df in groups.items():
    control = df[df.variant=="Control"]
    treatment = df[df.variant=="Treatment"]
    effect_estimate_se.append(get_effect_estimate_se(treatment.val, control.val))
    effect_estimate.append(get_effect_estimate(treatment.val, control.val))
    n.append(len(df))
    stratums.append(key)
    
strat_te = pd.DataFrame({
    "group": stratums,
    "effect_estimate_se": effect_estimate_se,
    "effect_estimate": effect_estimate,
    "n": n
})
strat_te["effect_estimate_se"] = strat_te["effect_estimate_se"]*strat_te["n"]/np.sum(strat_te["n"])
strat_te["effect_estimate"] = strat_te["effect_estimate"]*strat_te["n"]/np.sum(strat_te["n"])

strat_te = strat_te[["effect_estimate_se","effect_estimate","n"]].apply("sum")
print(strat_te)

res = (1-(strat_te.effect_estimate_se/normal_te.effect_estimate_se))*100 
print(f"""
    Сокращение дисперсии на {res[0]}%
""")
# =============================================================================
# =============================================================================
# CUPED
# =============================================================================
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.mean(df.loc[df.variant==1, 'revenue_after']) - np.mean(df.loc[df.variant==0, 'revenue_after'])
smf.ols('revenue_after ~ variant', data=df).fit().summary().tables[1]

theta = smf.ols('revenue_after ~ revenue_before', data=df).fit().params[1]
df['revenue_cuped'] = df['revenue_after'] - theta * (df['revenue_before'] - np.mean(df['revenue_before']))
smf.ols('revenue_cuped ~ variant', data=df).fit().summary().tables[1]
theta = df['revenue_before'].cov(df['revenue_after']) / df['revenue_before'].var() 
df['revenue_cuped'] = df['revenue_after'] - theta * (df['revenue_before'] - np.mean(df['revenue_before']))
smf.ols('revenue_cuped ~ variant', data=df).fit().summary().tables[1]

# =============================================================================
# =============================================================================
# Bootstrap
# =============================================================================
# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа
def get_bootstrap(
    data_0: list, # числовые значения первой выборки
    data_1: list, # числовые значения второй выборки
    boot_it: int = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    conf_level: float = 0.95, # уровень значимости,
    ba: bool = False
):
    boot_data = []
    for _ in tqdm(range(boot_it)): # извлекаем подвыборки
        boot_0 = data_0.sample(len(data_0), replace = True).values
        boot_1 = data_1.sample(len(data_1), replace = True).values
        boot_data.append(statistic(boot_0) - statistic(boot_1)) # mean() - применяем статистику
        
    # поправляем смещение
    if ba:
        orig_theta = statistic(data_0)-statistic(data_1) # разница в исходных данных
        boot_theta = np.mean(boot_data) # среднее по бутстрапированной разнице статистик
        delta_val = abs(orig_theta - boot_theta) # дельта для сдвига
        boot_data = [i - delta_val for i in boot_data] # сдвигаем бут разницу статистик, обратите внимание, что тут не вычитание
        print(f"""
            До бутстрапа: {orig_theta},
            После бутстрапа: {boot_theta},
            После коррекции: {np.mean(boot_data)}"""
        )

    left_quant = (1 - conf_level)/2
    right_quant = 1 - (1 - conf_level) / 2
    ci = pd.DataFrame(boot_data).quantile([left_quant, right_quant])

    # p-value
    p_1 = norm.cdf(x = 0, loc = np.mean(boot_data), scale = np.std(boot_data))
    p_2 = norm.cdf(x = 0, loc = -np.mean(boot_data), scale = np.std(boot_data))
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    plt.hist(pd.DataFrame(boot_data)[0], bins = 50)
    plt.style.use('ggplot')
    plt.vlines(ci,ymin=0,ymax=100,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "ci": ci, 
            "p_value": p_value}
# =============================================================================
# =============================================================================
# Bootstrap Ratio
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson, binom
from tqdm import tqdm

def bootstrap_ratio(df, variant_col, num_col, den_col, boot_it):
    is_test = df[variant_col] == 1
    rto_0 = sum(df[num_col][df[variant_col] == 0]) / sum(df[den_col][df[variant_col] == 0])
    rto_1 = sum(df[num_col][df[variant_col] == 1]) / sum(df[den_col][df[variant_col] == 1])
    delta = abs(rto_0 - rto_1)
    count = 0
    
    for i in range(boot_it):
        booted_index = is_test.sample(len(is_test)).values
        sample_0 = sum(df.loc[True == booted_index][num_col]) / sum(df.loc[True == booted_index][den_col])
        sample_1 = sum(df.loc[False == booted_index][num_col]) / sum(df.loc[False == booted_index][den_col])

        if abs(sample_0-sample_1)>=delta:
            count += 1
    pvalue = count / boot_it
    
    return pvalue


def bootstrap_ratio(df, variant_col, num_col, den_col, boot_it):
    is_test = df[variant_col] == 1
    rto_0 = sum(df[num_col][df[variant_col] == 0]) / sum(df[den_col][df[variant_col] == 0])
    rto_1 = sum(df[num_col][df[variant_col] == 1]) / sum(df[den_col][df[variant_col] == 1])
    delta = abs(rto_0 - rto_1)
    count = 0
    
    for i in range(boot_it):
        booted_index = is_test.sample(len(is_test)).values
        sample_0 = sum(df.loc[True == booted_index][num_col]) / sum(df.loc[True == booted_index][den_col])
        sample_1 = sum(df.loc[False == booted_index][num_col]) / sum(df.loc[False == booted_index][den_col])

        if abs(sample_0-sample_1)>=delta:
            count += 1
    pvalue = count / boot_it
    return pvalue


N = 10000
n_sim = 500
alpha = 0.5
pvalue_list = []

for i in tqdm(range(n_sim), desc="Simulation Progress"):
    if i % 100 == 0:
        print(i)
        
    df = pd.DataFrame({
        "variant": binom.rvs(1, 0.5, size=N), 
        "numerator": np.random.poisson(10, N),
        "denominator": np.random.poisson(5, N)
    })

    pvalue = bootstrap_ratio(df, "variant", "numerator", "denominator", 100)
    pvalue_list.append(pvalue)

print(sum(np.array(pvalue_list) < alpha) / n_sim)
sns.histplot(pvalue_list)
# =============================================================================
# =============================================================================
# Poisson Bootstrap
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson, binom

def poisson_bootstrap(num_0, den_0, num_1, den_1, n_bootstrap=2000):

    rto0 = np.array([num_0 / den_0])
    rto1 = np.array([num_1 / den_1])

    poisson_0 = poisson(1).rvs((n_bootstrap, rto0.size)).astype(np.int64)
    poisson_1 = poisson(1).rvs((n_bootstrap, rto1.size)).astype(np.int64)
    
    den_0 = np.array([den_0])
    den_1 = np.array([den_1])
    
    rto0 = np.matmul(rto0 * den_0, poisson_0.T)
    w0 = np.matmul(den_0, poisson_0.T)
    rto1 = np.matmul(rto1 * den_1, poisson_1.T)
    w1 = np.matmul(den_1, poisson_1.T)

    delta = rto1 / w1 - rto0 / w0
    positions = np.sum(delta < 0)

    pvalue = 2 * np.minimum(positions, n_bootstrap - positions) / n_bootstrap
    return pvalue

def bootstrap_poisson(rto0, w0, rto1, w1, boot_it=500):
    """
    Пуассоновский бутстрап
    :param rto0: Рассчитанное ratio для левого сплита
    :param w0: Знаменатель rto0
    :param rto1: Рассчитанное ratio для правого сплита
    :param w1: Знаменатель rto1
    :param boot_it: количество бут итераций
    :return: pvalue
    """
    
    poisson_0 = poisson(1).rvs((boot_it, rto0.size)).astype(np.int64)
    poisson_1 = poisson(1).rvs((boot_it, rto1.size)).astype(np.int64)

    rto1 = np.matmul(rto1 * w1, poisson_1.T)

    w1 = np.matmul(w1, poisson_1.T)

    rto0 = np.matmul(rto0 * w0, poisson_0.T)
    w0 = np.matmul(w0, poisson_0.T)
    
    delta = rto1 / w1 - rto0 / w0
    
    positions = np.sum(delta < 0)

    pvalue = 2 * np.minimum(positions, boot_it - positions) / boot_it
    return pvalue

n = 20000
df = pd.DataFrame({
    'session_cnt': np.random.randint(low = 1, high = 10, size = n),
    'revenue_amt': np.random.exponential(100, size=n),
    'split': np.random.randint(low = 0, high = 2, size = n)
})

display(df)

%%timeit -r 10
poisson_bootstrap(
    df[1:100].revenue_amt[df[1:100].split == 0],
    df[1:100].session_cnt[df[1:100].split == 0],
    df[1:100].revenue_amt[df[1:100].split == 1],
    df[1:100].session_cnt[df[1:100].split == 1]
)
# =============================================================================
# =============================================================================
# Delta Method
# =============================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind, poisson
from tqdm.auto import tqdm
import scipy
import matplotlib.pyplot as plt

def deltamethod(x_0, y_0, x_1, y_1):
    n_0 = y_0.shape[0]-1
    n_1 = y_0.shape[0]-1

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

def deltamethod_bucketed(df_0, df_1, num_col, den_col, n_col):
    x_0 = np.array(df_0[num_col])
    x_1 = np.array(df_1[num_col])
    y_0 = np.array(df_0[den_col])
    y_1 = np.array(df_1[den_col])
    n_0 = np.array(df_0[n_col].sum())
    n_1 = np.array(df_1[n_col].sum())

    mean_x_0 = x_0.sum() / n_0
    mean_x_1 = x_1.sum() / n_1
    mean_y_0 = y_0.sum() / n_0
    mean_y_1 = y_1.sum() / n_1
    
    var_x_0 = np.sum(np.array([abs(a - mean_x_0)**2 for a in x_0])) / n_0
    var_x_1 = np.sum(np.array([abs(a - mean_x_1)**2 for a in x_1])) / n_1
    var_y_0 = np.sum(np.array([abs(a - mean_y_0)**2 for a in y_0])) / n_0
    var_y_1 = np.sum(np.array([abs(a - mean_y_1)**2 for a in y_1])) / n_1
    cov_0 = np.sum((x_0 - mean_x_0.reshape(-1, 1)) * (y_0 - mean_y_0.reshape(-1, 1)), axis=1) / n_0
    cov_1 = np.sum((x_1 - mean_x_1.reshape(-1, 1)) * (y_1 - mean_y_1.reshape(-1, 1)), axis=1) / n_1

    var_0 = var_x_0 / mean_y_0 ** 2 + var_y_0 * mean_x_0 ** 2 / mean_y_0 ** 4 - 2 * mean_x_0 / mean_y_0 ** 3 * cov_0
    var_1 = var_x_1 / mean_y_1 ** 2 + var_y_1 * mean_x_1 ** 2 / mean_y_1 ** 4 - 2 * mean_x_1 / mean_y_1 ** 3 * cov_1

    rto_0 = np.sum(x_0) / np.sum(y_0)
    rto_1 = np.sum(x_1) / np.sum(y_1)
    statistic = (rto_1 - rto_0) / np.sqrt(var_0 / n_0 + var_1 / n_1)

    pvalue = 2 * np.minimum(norm(0, 1).cdf(statistic), 1 - norm(0, 1).cdf(statistic))
    return pvalue[0]

# =============================================================================
# =============================================================================
# Lineralization
# =============================================================================
def linearization(x_0, y_0, x_1, y_1):
    k = x_0.sum() / y_0.sum()
    l_0 = x_0 - k * y_0
    l_1 = x_1 - k * y_1
    return l_0, l_1


