import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm      
from matplotlib.widgets import Slider

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian

opts={"linewidth":3, "alpha":0.5}
figsize=(7,3)

def step(x, maxval=100, temperature=100):
  return maxval / (1.0 + np.exp(-temperature * x))

def noise(x):
  x = x*0.85 
  # np.random.seed(46)
  n_bands = 10
  ampl = np.random.normal(.05, .01, n_bands)
  freq = np.random.uniform(10, 30, n_bands)
  phase = np.random.uniform(0, 2*np.pi, n_bands)
  _noise = [ampl[i]*np.sin(2*np.pi*freq[i]*x + phase[i]) for i in range(len(ampl))] 
  _noise = np.sum(np.stack(_noise),axis=0)
  return _noise

def transmittance(x, delta, bound, height):
  def alpha(x, delta, bound, height):
    sigma = density(x,bound, height)
    return 1.0 - np.exp(-sigma * delta)
  return np.cumprod(1.0 - alpha(x, delta, bound, height))

def noisy_rect(x, bound, height):
  a = step(x+bound)
  b = step(-(x-bound))
  srect = np.minimum(a,b)
  return srect + srect*noise(x)*height

def density(x, bound, height):
  return noisy_rect(x, bound, height)

x = np.linspace(-.7, +.7, 1000)
delta = x[1] - x[0]

bound, height = 0.05, 0.1
font = {'family' : 'helvatica',
        'size'   : 6}

matplotlib.rc('font', **font)
figs, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 3))
fig = axes[0].plot(x, density(x, 0.35, 0.5), 'k', linewidth=opts["linewidth"]);
axes[0].tick_params(labelbottom=False)
axes[0].set_xticks([])
axes[0].set_title('Density')
Tl = transmittance(x, delta, bound, height)
Tr = np.flip(transmittance(-x, delta, bound, height))
fig1 = axes[1].plot(x, Tl, 'r', **opts)
fig2 = axes[1].plot(x, Tr, 'g', **opts)

axes[1].set_xticks([])
axes[1].tick_params(labelbottom=False)
axes[1].set_title('Transmittance (left/right)')

Sl = Tl * (1-np.exp(-density(x,bound,height))) 
Sr = Tr * (1-np.exp(-density(x,bound,height))) 
fig3 = axes[2].plot(x, Sl, 'r', **opts)
fig4 = axes[2].plot(x, Sr, 'g', **opts)
axes[2].tick_params(labelbottom=False)
axes[2].set_xticks([])
axes[2].set_title('View-dependent Surface (left/right)')

S = np.maximum(Sl, Sr)
S_tau = S>0.5
fig5 = axes[3].plot(x, S_tau, 'k', linewidth=opts["linewidth"])
axes[3].set_title('Thresholded Surface')

axes[3].tick_params(labelbottom=False)
axes[3].set_xticks([])
plt.tight_layout(pad=3.4, w_pad=0.5, h_pad=1.0)






axb = plt.axes([0.32, 0.05, 0.33, 0.03])
sb = Slider(axb, 'band', 0.05, 0.65, valinit=0.05)

axb2 = plt.axes([0.32, 0.01, 0.33, 0.03])
sb2 = Slider(axb2, 'noise', 0.1, 2, valinit=0.1)

def update(val):
    fig[0].set_data((x,density(x, val, sb2.val)))
    fig1[0].set_data((x,transmittance(x, delta, val, sb2.val)))
    fig2[0].set_data((x,np.flip(transmittance(-x, delta, val, sb2.val))))
    fig3[0].set_data((x,transmittance(x, delta, val, sb2.val)* (1-np.exp(-density(x,val,sb2.val))) ))
    fig4[0].set_data((x,np.flip(transmittance(-x, delta, val, sb2.val))* (1-np.exp(-density(x,val,sb2.val))) ))	
    Sl = transmittance(x, delta, val, sb2.val)* (1-np.exp(-density(x,val,sb2.val)))
    Sr = np.flip(transmittance(-x, delta, val, sb2.val))* (1-np.exp(-density(x,val,sb2.val)))
    S = np.maximum(Sl, Sr)
    S_tau = S>0.5
    fig5[0].set_data((x, S_tau))
sb.on_changed(update)

def update2(val):
    fig[0].set_data((x,density(x, sb.val, val)))
    fig1[0].set_data((x,transmittance(x, delta, sb.val, val)))
    fig2[0].set_data((x,np.flip(transmittance(-x, delta, sb.val, val))))
    fig3[0].set_data((x,transmittance(x, delta, sb.val, val)* (1-np.exp(-density(x,sb.val,val))) ))
    fig4[0].set_data((x,np.flip(transmittance(-x, delta, sb.val, val))* (1-np.exp(-density(x,sb.val,val))) ))	
    Sl = transmittance(x, delta, sb.val, val)* (1-np.exp(-density(x,sb.val,val)))
    Sr = np.flip(transmittance(-x, delta, sb.val, val))* (1-np.exp(-density(x,sb.val,val)))
    S = np.maximum(Sl, Sr)
    S_tau = S>0.5
    fig5[0].set_data((x, S_tau))
sb2.on_changed(update2)

plt.show()
