import numpy as np
import matplotlib.pyplot as plt
"""""
@author:Helia Hashemipour
"""""
plt.style.use('ggplot')
price_vec = np.load('btc_price.npy')
print('y =\n',price_vec)

plt.figure(figsize=(14,8))
plt.plot(list(range(len(price_vec))),price_vec,label='price')
plt.title('BTC Price 2h')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

price_vec = price_vec.reshape(-1,1)
n = price_vec.shape[0]

D = np.zeros((n-1 , n))
for i in range(n-1):
    D[i,i] = 1
    D[i,i+1] = -1
print('D =\n',D)


lambda_list = [0,10,100,1000,2500,5000,7500,10000]
x_list = []


for LAMBDA in lambda_list:
    # Ax = B --> min||Ax-b||^2
    A = np.vstack([np.eye(n), (LAMBDA**0.5)*D])
    b = np.vstack([price_vec,np.zeros((n-1,1))])

    # Least squares
    # x = INV(X.T* X) * X.T * b
    x = np.matmul(np.linalg.inv(np.matmul(A.T,A)),  np.matmul(A.T,b))
    x_list.append(x)
print('min||Ax-b||^2\nA =\n',A)
print('b =\n',b)

plt.figure(figsize=(12,8))
plt.plot(list(range(len(price_vec))),price_vec,label='price')
for ind in range(len(lambda_list)):
    plt.plot(list(range(len(price_vec))),x_list[ind] ,label=f'Denoised lambda={lambda_list[ind]}',lw=2.5)
plt.title('Denoising BTC Price with LS')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()


plt.figure(figsize=(15,12))
for ind in range(len(lambda_list)):
    plt.subplot(4,2,ind+1)
    plt.plot(list(range(len(price_vec))),price_vec,label='price')
    plt.plot(list(range(len(price_vec))),x_list[ind] ,label=f'Denoised',lw=3)
    plt.title(f'lambda = {lambda_list[ind]}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

plt.tight_layout(pad=1)
plt.show()





