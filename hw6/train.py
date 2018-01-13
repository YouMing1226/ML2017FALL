from skimage import io, transform
from numpy import linalg as la
from PIL import Image
import numpy as np 
import os, sys

small_img = []

print('Loading images...')
all_img_path = os.path.join(sys.argv[1],'*.jpg')
img_collection = io.imread_collection(all_img_path)

for img in img_collection:
    #small_tmp = transform.resize(img,(img.shape[0]/2, img.shape[1]/2))
    small_tmp = img.flatten()
    small_img.append(small_tmp)

img = np.array(small_img)
avg = np.average(img, axis=0)
small_img = img - avg 
small_img = (np.array(small_img)).T
print('Image shape:',small_img.shape)

print('PCA...')
U, sigma, VT = la.svd(small_img, full_matrices=False)

print('U:',U.shape)
print('Sigma:',sigma.shape)
print('VT:',VT.shape)

weight_1 = np.around(sigma[0]*100/np.sum(sigma),decimals=1)
weight_2 = np.around(sigma[1]*100/np.sum(sigma),decimals=1)
weight_3 = np.around(sigma[2]*100/np.sum(sigma),decimals=1)
weight_4 = np.around(sigma[3]*100/np.sum(sigma),decimals=1)

print('Weight 1: {}%'.format(weight_1))
print('Weight 2: {}%'.format(weight_2))
print('Weight 3: {}%'.format(weight_3))
print('Weight 4: {}%'.format(weight_4))

img_index = int((sys.argv[2]).split('.jpg')[0])
k_eigenface = 4

def rec_img(image, U):
    print('Reconstruction image',img_index)
    weight = []
    image = image - avg

    for k in range(k_eigenface):
        weight.append(np.dot(image, U.T[k]))
  
    U=U.T
    #test = np.array(-U[9])

    for i in range(k_eigenface):
        if i == 0:
            M = U[i]*weight[i]
        else:
            M += U[i]*weight[i]

    M += avg
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)

    M = np.reshape(M, (600, 600, 3))
    #im = Image.fromarray(M)
    print('Saving image to ./reconstruct_{}.jpg'.format(str(img_index)))
    io.imsave('./reconstruct_'+str(img_index)+'.jpg', M)

rec_img(img[img_index], U)

#avg -= np.min(avg)
#avg /= np.max(avg)
#avg = (avg*255).astype(np.uint8)
#X = np.reshape(avg, (600, 600, 3))

#x = Image.fromarray(X)
#x.save('./avg_face.jpg')



