import numpy as np
import matplotlib.pyplot as plt

testY = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_testY.csv", delimiter=',')
pred3 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch3.csv", delimiter=',')
pred5 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch5.csv", delimiter=',')
pred8 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch8.csv", delimiter=',')
pred10 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch10.csv", delimiter=',')

#테스트셋 다른차량
pred3_1 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch3-1.csv", delimiter=',')
pred5_1 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch5-1.csv", delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(testY, label='True',color='red')
ax.plot(pred3_1, label='3epoch_Prediction1',color='blue')
ax.plot(pred5_1, label='5epoch_Prediction1',color='green')
# ax.plot(pred8, label='8epoch_Prediction',color='yellow')
# ax.plot(pred10, label='10epoch_Prediction',color='black')
ax.grid()
ax.legend()
plt.show()