import matplotlib.pyplot as plt
plt.style.use('ggplot')

#QQPLOT
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(x, color="green")
plt.title("Normal Q-Q Plot")
plt.show()

# MDE VS Sample Size
viz = sns.lineplot(x=sample_sizes, y=effects)
viz.set_xlabel("Sample Size")
viz.set_ylabel("Effect Amount")