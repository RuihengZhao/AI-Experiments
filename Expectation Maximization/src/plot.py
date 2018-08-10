from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

delta = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8]

meanAccuracyBefore = [0.74, 0.742, 0.744, 0.712, 0.667, 0.737, 0.6, 0.528, 0.59, 0.589, 0.519, 0.622, 0.493, 0.454, 0.414, 0.438, 0.483, 0.452, 0.45, 0.475]
meanAccuracyAfter = [0.93, 0.928, 0.927, 0.928, 0.926, 0.927, 0.925, 0.927, 0.927, 0.926, 0.926, 0.925, 0.915, 0.917, 0.898, 0.896, 0.885, 0.876, 0.865, 0.859]

stdDeviationBefore = [0, 0.057, 0.093, 0.131, 0.15, 0.054, 0.135, 0.183, 0.129, 0.151, 0.209, 0.178, 0.17, 0.182, 0.196, 0.177, 0.181, 0.176, 0.187, 0.208]
stdDeviationAfter = [0, 0.004, 0.005, 0.004, 0.064, 0.004, 0.005, 0.004, 0.005, 0.14, 0.226, 0.004, 0.218, 0.237, 0.263, 0.166, 0.236, 0.255, 0.205, 0.247]

x = np.array(delta)
y = np.array(meanAccuracyBefore)
y2 = np.array(meanAccuracyAfter)

e = np.array(stdDeviationBefore)
e2 = np.array(stdDeviationAfter)

# plt.errorbar(x, y, e, linestyle='solid', marker='^')
# plt.errorbar(x, y2, e2, linestyle='solid', marker='^')

# plt.show()
plt.plot(x, y)
plt.errorbar(x, y, yerr = e, fmt='o')
plt.errorbar(x, y2)
plt.errorbar(x, y2, yerr = e2, fmt='-')

plt.show()