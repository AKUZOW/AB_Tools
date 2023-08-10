def get_se(x):
    """
    Function to calculate the Standart error for an avr/proportion in the array-like data structure
    :param x: an array to calculate the standart error for
    :return: Standart error
    """
    return np.std(x) / np.sqrt(len(x))

def get_ci_95(x):
    ci_upper = np.mean(x) + 1.96*get_se(x)
    ci_lower = np.mean(x) - 1.96*get_se(x)
    return {"ci_lower": ci_lower,
            "ci_upper": ci_upper}
# =============================================================================
# Ztest
# =============================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats import weightstats


std = 1
# alpha = 0.99
mu_x = np.random.uniform(-1,1) 
mu_y = np.random.uniform(-1,1)

x = norm.rvs(size = 100, loc = mu_x, scale = std)
y = norm.rvs(size = 100, loc = mu_y, scale = std)

res = weightstats.ztest(x, y, alternative='two-sided', value=mu_x-mu_y, ddof=1.0)
print("Two-sample z-Test")
print("Z-statistic:", res[0])
print("P-value:", res[1])

# Proportions
import random
import numpy as np
from scipy.stats import uniform, binom, norm
import statsmodels.stats as sm

n = random.choice([i for i in range(490,510+1)])
m = random.choice([i for i in range(490,510+1)])

p_x = uniform.rvs(size=1, loc=0, scale=1)
p_y = uniform.rvs(size=1, loc=0, scale=1) 

x = binom.rvs(n=1, p=p_x, size=n)
y = binom.rvs(n=1, p=p_y, size=m)

stat, pval = sm.proportion.proportions_ztest([sum(x), sum(y)], 
                                             [len(x), len(y)]) 

print("2-sample test for equality of proportions")
print("test statistic =", stat)
print("p-value =", pval)

# C.I.
statsmodels.stats.weightstats.zconfint(x1, x2=None, value=0, alpha=0.05, alternative='two-sided', usevar='pooled', ddof=1.0)
# =============================================================================
# =============================================================================
# Ttest
# =============================================================================
import numpy as np
from scipy.stats import norm, uniform
import statsmodels.api as sm


std = 1
n = 100

min_ = -0.5
max_ = 0.5

mu_x = uniform.rvs(size=1, loc=min_, scale=max_-min_)
mu_y = uniform.rvs(size=1, loc=min_, scale=max_-min_) 

x = norm.rvs(size = n, loc = mu_x, scale = std)
y = norm.rvs(size = n, loc = mu_y, scale = std)

res = sm.stats.ttest_ind(x, y)
res = sm.stats.ttest_ind(x, y, usevar="pooled") 
print("Two Sample t-test")
print("t =", res[0])
print("p-value =", res[1])
print("df = ", res[2])

# C.I.
statsmodels.stats.weightstats._tconfint_generic(mean, std_mean, dof, alpha, alternative)
# =============================================================================
# =============================================================================
# Kstest
# =============================================================================
import numpy as np
from scipy.stats import uniform, binom, norm, kstest

x = norm.rvs(size=1500, loc = 0, scale = 1)

kstest(x, "norm")
kstest(x, "expon")
kstest(x, "binom", args=(1500, 0.16))
# =============================================================================
# =============================================================================
# SW Test
# =============================================================================
import numpy as np
from scipy.stats import norm, shapiro

x = norm.rvs(size = 1500, loc = 0, scale = 1)
W, p_val = shapiro(x)
print("Shapiro-Wilk normality test")
print("W =", W)
print("P-value =", p_val)
# =============================================================================
# =============================================================================
# Barlett Test
# =============================================================================
from scipy.stats import norm, bartlett
import pandas as pd
import numpy as np

std_x = 1
std_y = 1.57
std_z = 1.71

mu_x = 1.17
mu_y = 1.21
mu_z = 3.51

n = 250

x = norm.rvs(size = n, loc = mu_x, scale = std_x)
y = norm.rvs(size = n, loc = mu_y, scale = std_y)
z = norm.rvs(size = n, loc = mu_z, scale = std_z)

df = pd.DataFrame({"A": x, "B": y, "C":z})

bartlett(df.A, df.B, df.C)
# =============================================================================
# =============================================================================
# Levene Test
# =============================================================================
from scipy.stats import norm, levene
import numpy as np
import pandas as pd

std_x = 1
std_y = 1.57

n_x = 200
n_y = 250

x = norm.rvs(size = n_x, loc = 0, scale = std_x)
y = norm.rvs(size = n_y, loc = 0, scale = std_y)

levene(x, y, center = "mean")
levene(x, y, center = "median")
levene(x, y, center = "trimmed")
# =============================================================================
# =============================================================================
# MW-U Test
# =============================================================================
from scipy.stats import norm, binom, expon, wilcoxon
import numpy as np
import statsmodels.api as sm

x = norm.rvs(size = 500, loc = 0, scale = 1)
y = norm.rvs(size = 500, loc = 1, scale = 1)

wilcoxon(x, y)

x = norm.rvs(size = 500, loc = 0, scale = 1)
y = norm.rvs(size = 500, loc = 0, scale = 50)
wilcoxon(x, y)

x = binom.rvs(size=300, n=1, p=0.18)
y = expon.rvs(size=300, scale=0.2)
print(np.mean(x))
print(np.mean(y))

wilcoxon(x, y)


res = sm.stats.ttest_ind(x, y)
print("Two Sample t-test")
print("t =", res[0])
print("p-value =", res[1])
print("df = ", res[2])