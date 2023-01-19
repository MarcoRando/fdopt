
from scipy.fftpack import fft as spfft
from scipy.fftpack import dct as spdct
from scipy.fftpack import dst as spdst
import numpy as np 

sq2 = np.sqrt(2)
W2  = np.array([[1,1],[1,-1]])
H2  = W2/sq2

mat2mul = lambda M,v: [M[0][0]*v[0]+M[0][1]*v[1],M[1][0]*v[0]+M[1][1]*v[1]]
# mat2mul = lambda M,v: np.matmul(M,v)
hf1 = lambda data: np.array([mat2mul(H2,v2) for v2 in np.array(data).reshape(len(data)>>1,2)]).transpose().flatten()
hf1I= lambda data: np.array([mat2mul(H2,v2) for v2 in np.array(data).reshape(2,len(data)>>1).transpose()]).flatten() 


#"Haar gives the (fast) Haar transform
#  of the list x, if x has length 2^k."
def Haar(data):
    d1  = 1.0*np.array(data)
    k1,k2 = 1,len(data)
    while k2 > 1:
        d1 = d1.reshape(k1,k2)
        d1[0] = hf1(d1[0])
        k1 <<= 1
        k2 >>= 1
    return d1#.flatten()


Hartley = lambda x : np.array([ np.real(el) for el in spfft(np.array(x))*(1-1j)])/np.sqrt(len(x))


DTC1 = lambda x: spdct(x,type=1,norm='ortho')
DTC2 = lambda x: spdct(x,type=2,norm='ortho')
DTC3 = lambda x: spdct(x,type=3,norm='ortho')
DTC4 = lambda x: spdct(x,type=4,norm='ortho')

#"DST1 computes an orthogonal version of the 
# DST-I transform."
DST1 = lambda x: spdst(x,type=1,norm='ortho')
DST4 = lambda x: spdst(x,type=4,norm='ortho')

def DST2(x):
    d2 = spdst(x,type=2,norm='ortho')
    d2[ 0] *= sq2
    d2[-1] /= sq2
    return d2
def DST3(x):
    d3 = np.array(x)/sq2
    d3[-1] *= sq2
    d3 = spdst(d3,type=3,norm=None)
    return d3/np.sqrt(len(x))

#"DPCA computes the orthogonal transform 
# corresponding to PCA."
#"PCA computes the PCA of x."

PCAinter  = lambda x: np.append(np.array([np.zeros(len(x)),np.array(x)]).transpose().flatten(),0)
DPCA      = lambda x: DCT3(PCAinter(x))[:len(x)] *sq2
scaledSum = lambda x: np.cumsum(np.array(x))/np.sqrt(len(x))
PCA       = lambda x: scaledSum(DPCA(x))


Fwd = lambda x:x