import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os


def save_reshaped_cir(input_file, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mat = scipy.io.loadmat(input_file)
    cirmat = mat['cirmat']  
    norm_cir = cirmat / np.linalg.norm(cirmat, axis=1, keepdims=True)

    chan1_real = np.real(norm_cir)
    chan2_img = np.imag(norm_cir)
    cir_channels = np.stack((chan1_real, chan2_img), axis=0)  
    cir_channels = cir_channels.transpose(1, 0, 2) 

    pad_width = 49*49 - cir_channels.shape[2] 
    cir_padded = np.pad(cir_channels, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

    cir_reshaped = cir_padded.reshape(187, 2, 49, 49)
    np.save(out_file, cir_reshaped)

def plot_cir(cir):
  """
  Plot the channel impulse response.
  """
  y = np.abs(cir)
  plt.figure(figsize=(10, 5))
  plt.plot(y, label='Magnitude')
  plt.title('Channel Impulse Response')
  plt.xlabel('Sample Index')
  plt.ylabel('Magnitude')
  plt.grid()
  plt.legend()
  plt.show()