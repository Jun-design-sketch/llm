import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

K=50
D=64
pos_enc=np.empty((K,D))

for i in range(K):
  for k in range(D//2):
    theta=i/(10000 ** (2*k/D))
    pos_enc[i,2*k]=np.sin(theta)
    pos_enc[i,2*k+1]=np.cos(theta)

# im=plt.imshow(pos_enc)
# plt.xlabel("dimension")
# plt.ylabel("position")
# plt.colorbar(im)
# plt.show()

dot_matrix=np.matmul(pos_enc,pos_enc.T)
im=plt.imshow(dot_matrix,origin="lower")
plt.xlabel("position")
plt.ylabel("position")
plt.colorbar(im)
plt.show()
