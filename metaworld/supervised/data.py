import torch
from torch.utils.data import Dataset
import string
translator = str.maketrans('', '', string.punctuation)
import random
import glob
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle

FRAME_DIR = '../PPO-PyTorch/metaworld-dataset/'
OUTPUT_DIR = '../PPO-PyTorch/metaworld-dataset/processed-frames'
OUTPUT_DIR2 = '../PPO-PyTorch/metaworld-dataset-bad/processed-frames'

class PadBatch:
  def __init__(self):
    pass

  def __call__(self, batch):
    traj_batch, lang_batch, traj_len_batch, lang_len_batch, labels_batch, obj_batch, \
        env_batch = zip(*batch)
    traj_batch = pad_sequence(traj_batch, batch_first=True)
    lang_batch = pad_sequence(lang_batch, batch_first=True)
    lang_batch = torch.from_numpy(np.array(lang_batch))
    traj_len_batch = torch.Tensor(traj_len_batch)
    lang_len_batch = torch.Tensor(lang_len_batch)
    return traj_batch, lang_batch, traj_len_batch, lang_len_batch, \
        labels_batch, obj_batch, env_batch

class Data(Dataset):
  def __init__(self, mode, sampling, prefix, repeat=1):
    self.sampling = sampling
    self.prefix = prefix
    self.vocab = pickle.load(open('vocab.pkl', 'rb'))
    self.descriptions = self.load_descriptions(mode)
    if mode == 'train':
      self.video_ids = list(range(80))
    elif mode == 'valid':
      self.video_ids = list(range(80, 100))
    else:
      raise NotImplementedError('Invalid mode!')
    self.N_OBJ = 13
    self.N_ENV = len(self.video_ids)
    self.thresh = 0.
    self.repeat = repeat

  def __len__(self):
    return 2 * self.N_OBJ * self.N_ENV * self.repeat

  def set_thresh(self, thresh):
    self.thresh = thresh

  def encode_description(self, descr):
    return torch.from_numpy(np.array([self.vocab.index(word) for word in descr.split()]))

  def load_descriptions(self, mode):
    result = {}
    filename = '{}.txt'.format(mode)
    with open(filename) as f:
      for line in f.readlines():
        line = line.strip()
        vidid, _, descr = line.split('\t')
        descr = descr.translate(translator).lower()
        descr_enc = self.encode_description(descr)
        vidid = eval(vidid)
        if vidid in result:
          descr_list = result[vidid]
        else:
          descr_list = []
        result[vidid] = descr_list + [descr_enc]
    return result

  def load_env_objects(self, obj, env):
    result = []
    with open('../PPO-PyTorch/metaworld-dataset/obj{}-env{}/env.txt'.format(obj, env)) as f:
      for line in f.readlines():
        line = line.strip()
        parts = line.split()
        result.append(eval(parts[-1]))
    return result

  def __getitem__(self, index):
    if index >= len(self) // 2:
      label = 0
      index -= len(self) // 2
    else:
      label = 1

    obj = index // (self.repeat * self.N_ENV)
    env = self.video_ids[index % self.N_ENV]
    env_objects = self.load_env_objects(obj, env)

    frames = torch.from_numpy(torch.load(open('{}/obj{}-env{}-50x50.pt'.format(OUTPUT_DIR, obj, env), 'rb')))

    if self.prefix:
      t = np.random.randint(1, len(frames)+1)
      frames = frames[:t]

    if self.sampling == 'fixed':
      frames = frames[::10]
    elif self.sampling == 'random':
      while True:
        selected = np.random.random(len(frames)) > 0.8
        if np.sum(selected) > 0:
          break
      frames = frames[selected]
      
    if label == 1:
      t = np.random.randint(0, len(self.descriptions[obj]))
      descr = self.descriptions[obj][t]
    else:
      tt = np.random.random()
      if tt < self.thresh:
        # alternate traj
        frames = torch.from_numpy(torch.load(open('{}/obj{}-env{}-50x50.pt'.format(OUTPUT_DIR2, obj, env), 'rb')))
        t = np.random.randint(1, len(frames)+1)
        frames = frames[:t:2]
        t = np.random.randint(0, len(self.descriptions[obj]))
        descr = self.descriptions[obj][t]
      else:
        # alternate lang
        alt_obj = env_objects[1:]
        if len(alt_obj) == 0:
          alt_obj = list(range(0,obj)) + list(range(obj+1,self.N_OBJ))
        obj_ = np.random.choice(alt_obj)
        t = np.random.randint(0, len(self.descriptions[obj_])-1)
        descr = self.descriptions[obj_][t]
    return frames, descr, len(frames), len(descr), label, obj, env

