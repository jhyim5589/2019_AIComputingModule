import numpy as np
import matplotlib.pyplot as plt

# ground truth로 testY, 예측값으로 pred5를 불러온 뒤 plot
testY = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_testY.csv", delimiter=',')
pred5 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch5.csv", delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(testY, label='True',color='red')
ax.plot(pred5, label='5epoch_Prediction1',color='green')

ax.grid()
ax.legend()
plt.show()
