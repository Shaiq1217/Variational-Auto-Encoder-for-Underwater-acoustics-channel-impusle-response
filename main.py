from src.cir import save_reshaped_cir, plot_cir
import glob
import os
import numpy as np

def preprocess_data():
    # Saving reshaped CIR
  mat_files = glob.glob("data\\cirmat\\*.mat")
  out_path = "data\\cir\\"
  for idx, cir in enumerate(mat_files):
    out_file = os.path.join(out_path, f"cir_{idx}.npy")
    save_reshaped_cir(cir, out_file)
    print(f"[✓] Saved: {out_file}")

def load_data():
  data_load = glob.glob(os.path.join("data", "cir", "*.npy"))
  cir_data = np.concatenate([np.load(cir_name) for cir_name in data_load], axis = 0)
    
  print(f"[✓] Loaded {len(data_load)} CIR files. Shape: {cir_data.shape}")
    

def main():
  load_data()

  

if __name__ == "__main__":
  main()