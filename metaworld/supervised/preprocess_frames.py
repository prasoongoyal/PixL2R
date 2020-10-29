import torch
import numpy as np
from PIL import Image
import glob

FRAME_DIR = '../PPO-PyTorch/metaworld-dataset/'
OUTPUT_DIR = '../PPO-PyTorch/metaworld-dataset/processed-frames'
N_OBJ = 13
N_ENV = 100

def main():
  for obj in range(13):
    for env in range(100):
      result = []
      img_files = glob.glob('{}/obj{}-env{}/right*.png'.format(FRAME_DIR, obj, env))
      # img_files = img_files[::-1][::5][::-1]
      for f in img_files:
        img = Image.open(f)
        img = img.resize((50, 50))
        img = np.array(img)
        result.append(img)
      result = np.transpose(result, (0, 3, 1, 2)) / 255.
      torch.save(result, open('{}/obj{}-env{}-50x50.pt'.format(OUTPUT_DIR, obj, env), 'wb'))
  # return torch.from_numpy(result)

if __name__ == '__main__':
  main()
