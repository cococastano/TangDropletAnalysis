import scipy.stats as stats

UW = [0.914, 0.8262, 0.654, 0.905, 0.68, 0.926, 0.892, 0.886, 0.566, 0.847, 0.92, 0.865, 0.891, 0.611, 0.396]

W = [0.82, 0.811, 0.659, 0.811, 0.593, 0.743, 0.347, 0.542, 0.209, 0.174, 0.71, 0.68, 0.611, 0.374, 0.366]

print(len(UW), len(W))

f, p = stats.f_oneway(UW, W)

print('Results of One-way ANOVA')
print('F-value: ', f)
print('P-value: ', p)