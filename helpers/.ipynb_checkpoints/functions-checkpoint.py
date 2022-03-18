import tensorflow.keras.backend as K
import torch
import torch.nn.functional as F
from scipy.interpolate import interp2d
from tqdm import tqdm
import numpy as np
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)


def vae_loss(recon_x, x, mu, logvar):
	bce = F.binary_cross_entropy(recon_x, x, size_average=False)

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
	return bce + kld

def get_assymetry(data, ps, points, orthog=False):
    # асимметрия ливня вдоль и поперек направнения наклона
    first = True
    assym_res = []
    for i in tqdm(range(len(data))):
        img = data[i]
        p = ps[i]
        #print('momentum', p)
        point = points[i, :2]
#        zoff = 50
        zoff = 25
        point0 = point[0] + zoff*p[0]/p[2]
        point1 = point[1] + zoff*p[1]/p[2]
    
        if orthog:
            line_func = lambda x: (x - point0) / p[0] * p[1] + point1
        else:
            line_func = lambda x: -(x - point0) / p[1] * p[0] + point1
    
    
    
        x = np.linspace(-14.5, 14.5, 30)
        y = np.linspace(-14.5, 14.5, 30)

        xx, yy = np.meshgrid(x, y)
    
        idx = np.where(yy - line_func(xx) < 0)
        if (not orthog and p[1]<0):
            idx = np.where(yy - line_func(xx) > 0)
    
        zz = np.ones((30, 30))
        zz[idx] = 0
    
        assym = (np.sum(img * zz) - np.sum(img * (1 - zz))) / np.sum(img)
        assym_res.append(assym)
        
#        if (first and assym > 0.999):
#            print ("Iteration ", i)
#            print ("Momentum :", ps[i])
#            print ("Point: ", points[i])
#            print ("Correction: ", 50.*p[0]/p[2], 50.*p[1]/p[2])
#            print ("Corrected: ", point0, point1)
#            plt.imshow(np.log10(img))
#            first = False
#            break
    return assym_res

def get_shower_width(data, ps, points, orthog=False):
    
    # ширина ливня вдоль и поперек направления
    
    res = []
    spreads = []
    
    for i in tqdm(range(len(data))):
        
        img = data[i]
        p = ps[i]
        point = points[i]
        zoff = 25
        point0 = point[0] + zoff*p[0]/p[2]
        point1 = point[1] + zoff*p[1]/p[2]
        
        if orthog:
            line_func = lambda x: -(x - point0) / p[0] * p[1] + point1
        else:
            line_func = lambda x:  (x - point0) / p[1] * p[0] + point1
    
        x = np.linspace(-14.5, 14.5, 30)
        y = np.linspace(-14.5, 14.5, 30)

        bb = interp2d(x, y, img, kind='cubic')

        x_ = np.linspace(-14.5, 14.5, 100)

        y_ = line_func(x_)
        
        rescale = np.sqrt(1+(p[1]/p[0])*(p[1]/p[0]))

#        vals = []
        sum0 = 0
        sum1 = 0
        sum2 = 0
        for i in range(100):
#            vals.append(bb(x_[i], y_[i]))
            ww = bb(x_[i], y_[i])
            if ww < 0: ww = 0
            sum0 += ww
            sum1 += rescale*x_[i]*ww
            sum2 += (rescale*x_[i])*(rescale*x_[i])*ww
            
#        vals_ = np.std(vals)
        sum1 = sum1/sum0
        sum2 = sum2/sum0
        if sum2 >= sum1*sum1 :
            sigma = np.sqrt (sum2 - sum1*sum1)
            spreads.append(sigma[0])
        else:
#            print sum2, sum1*sum1, sum2-sum1*sum1 
            spreads.append(0)
        
    return spreads