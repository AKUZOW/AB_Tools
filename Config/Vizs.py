import matplotlib.pyplot as plt
plt.style.use('ggplot')

#QQPLOT
import statsmodels.api as sm
import matplotlib.pyplot as plt
fig = sm.qqplot(x, color="green")
plt.title("Normal Q-Q Plot")
plt.show()

# MDE VS Sample Size
viz = sns.lineplot(x=sample_sizes, y=effects)
viz.set_xlabel("Sample Size")
viz.set_ylabel("Effect Amount")

# 95% C.I. plot
plt.hist(pd.Series(median_boot_data), bins = 50)
plt.style.use('ggplot')
plt.vlines(ci_median,ymin=0,ymax=300,linestyle='--',colors='black')
plt.xlabel('boot_data')
plt.ylabel('frequency')
plt.title("Histogram of boot_data")
plt.show()