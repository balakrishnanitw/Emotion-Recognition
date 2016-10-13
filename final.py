import glob
import numpy as np
#import struct
from scipy.io.wavfile import read
from sklearn import preprocessing
wavs = []
mean_all = []
maxima_all = []
minima_all = []
rms_all = []
zcr_all = []
lpc_all = []
for filename in glob.glob('*.wav'):
    #print(filename)
    wavs.append(read(filename))
    signal = read(filename)
    signal2array= np.array(signal[1], dtype=float)  
    mean=np.mean(signal2array)
    mean_all.append(mean)
    maxima=signal2array.max()
    maxima_all.append(maxima)
    minima = signal2array.min()
    minima_all.append(minima)
    from numpy import mean, sqrt, square
    rms = sqrt(mean(square(signal2array)))
    rms_all.append(rms)
    zcr=(np.diff(np.sign(signal2array)) != 0).sum()
    #array_size=signal2array.size
    #zcr = count/array_size 
    zcr_all.append(zcr)


def lpc(signal, order, axis=-1):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Returns
    -------
    a : array-like
        the solution of the inversion.
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.

    Notes
    -----
    This uses Levinson-Durbin recursion for the autocorrelation matrix
    inversion, and fft for the autocorrelation computation.

    For small order, particularly if order << signal size, direct computation
    of the autocorrelation is faster: use levinson and correlate in this case."""
    n = signal.shape[axis]
    if order > n:
        raise ValueError("Input signal must have length >= order")

    #r = acorr_lpc(signal, axis)
    return levinson_1d(signal2array, order)
def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

   # if not np.isreal(r[0]):
    ##    raise ValueError("First item of input must be real.")
    #elif not np.isfinite(1/r[0]):
     #   raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in xrange(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]
           # print a
        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k
    
LPCcoeff = lpc(signal2array,20) 
lpc_all.append(LPCcoeff)
#b=list(lpc_all)
#l=b.tolist()
#LPCcoeff.tolist()

mean_scaled= preprocessing.scale(mean_all)
#q=mean_normalized.mean()
#w=X_scaled.std()
mean_reshaped=mean_scaled.reshape(1,-1)
#normalizer = preprocessing.Normalizer().fit(u)
mean_normalized = preprocessing.normalize(mean_reshaped,norm='l2')


maxima_scaled= preprocessing.scale(maxima_all)
maxima_reshaped=maxima_scaled.reshape(1,-1)
maxima_normalized = preprocessing.normalize(maxima_reshaped,norm='l2')

minima_scaled= preprocessing.scale(maxima_all)
minima_reshaped=maxima_scaled.reshape(1,-1)
minima_normalized = preprocessing.normalize(minima_reshaped,norm='l2')

rms_scaled= preprocessing.scale(rms_all)
rms_reshaped=rms_scaled.reshape(1,-1)
rms_normalized = preprocessing.normalize(rms_reshaped,norm='l2')

zcr_scaled= preprocessing.scale(zcr_all)
zcr_reshaped=rms_scaled.reshape(1,-1)
zcr_normalized = preprocessing.normalize(zcr_reshaped,norm='l2')

#lpc_scaled= preprocessing.scale(lpc_all)
#lpc_reshaped=lpc_scaled.reshape(1,-1)
#lpc_normalized = preprocessing.normalize(lpc_reshaped,norm='l2')
list_all= list(zip(mean_normalized,maxima_normalized,minima_normalized,rms_normalized))
