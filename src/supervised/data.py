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

DATA_DIR = '../../data/'
FRAMES_DIR = '../../data/processed-frames/'
ENVS_DIR = '../../data/envs/'

TRAIN_VIDEOS = list(range(80))
VALID_VIDEOS = list(range(80, 100))
TEST_VIDEOS = [50, 92, 95]

class PadBatch:
    def __init__(self):
        pass

    def __call__(self, batch):
        traj_right_batch, traj_left_batch, traj_center_batch, \
        lang_batch, lang_enc_batch, traj_len_batch, lang_len_batch, labels_batch, obj_batch, \
                env_batch, weight_batch = zip(*batch)
        traj_right_batch = pad_sequence(traj_right_batch, batch_first=True)
        traj_left_batch = pad_sequence(traj_left_batch, batch_first=True)
        traj_center_batch = pad_sequence(traj_center_batch, batch_first=True)
        lang_enc_batch = pad_sequence(lang_enc_batch, batch_first=True)
        lang_enc_batch = torch.from_numpy(np.array(lang_enc_batch))
        traj_len_batch = torch.Tensor(traj_len_batch)
        lang_len_batch = torch.Tensor(lang_len_batch)
        weight_batch = torch.Tensor(weight_batch)
        labels_batch = torch.Tensor(labels_batch)
        return traj_right_batch, traj_left_batch, traj_center_batch, \
                lang_batch, lang_enc_batch, traj_len_batch, lang_len_batch, \
                labels_batch, obj_batch, env_batch, weight_batch

class Data(Dataset):
    def __init__(self, mode, repeat=10):
        self.vocab = pickle.load(open('{}/vocab_train.pkl'.format(DATA_DIR), 'rb'))
        self.descriptions = self.load_descriptions(mode)
        self.video_ids = self.get_video_ids(mode)
        self.N_OBJ = 13
        self.N_ENV = len(self.video_ids)
        self.repeat = repeat

    def get_video_ids(self, mode):
        if mode == 'train':
            video_ids = TRAIN_VIDEOS
        elif mode == 'valid':
            video_ids = VALID_VIDEOS
        else:
            raise NotImplementedError('Invalid mode!')
        for vid in TEST_VIDEOS:
            try:
                video_ids.remove(vid)
            except ValueError:
                pass
        return video_ids

    def __len__(self):
        return 2 * self.N_OBJ * self.N_ENV * self.repeat

    def encode_description(self, descr):
        result = []
        for w in descr.split():
                try:
                        t = self.vocab.index(w)
                except ValueError:
                        t = self.vocab.index('<unk>')
                result.append(t)
        return torch.Tensor(result)

    def load_descriptions(self, mode):
        descriptions = pickle.load(open('{}/{}_descr.pkl'.format(DATA_DIR, mode), 'rb'))
        result = {}
        for i in descriptions.keys():
                descr_list = descriptions[i]
                result[i] = [(d, self.encode_description(d)) for d in descr_list]
        return result

    def load_env_objects(self, obj, env):
        result = []
        with open('{}/obj{}-env{}.txt'.format(ENVS_DIR, obj, env)) as f:
            for line in f.readlines():
                line = line.replace('(', '').replace(',', '').replace(')', '')
                parts = line.split()
                x = eval(parts[0])
                y = eval(parts[1])
                obj = eval(parts[2])
                result.append(obj)
        return result

    def load_frames(self, obj, env):
        frames_r = torch.from_numpy(torch.load(open('{}/obj{}-env{}-right-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))
        frames_l = torch.from_numpy(torch.load(open('{}/obj{}-env{}-left-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))
        frames_c = torch.from_numpy(torch.load(open('{}/obj{}-env{}-center-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))
        n_frames_total = len(frames_r)

        n_frames = np.random.randint(1, (n_frames_total+1))
        weight = (n_frames / n_frames_total)
        frames_r = frames_r[:n_frames]
        frames_l = frames_l[:n_frames]
        frames_c = frames_c[:n_frames]

        while True:
            selected = np.random.random(n_frames) > 0.9
            if np.sum(selected) > 0:
                break
        frames_r = frames_r[selected]
        frames_l = frames_l[selected]
        frames_c = frames_c[selected]
            
        return frames_r, frames_l, frames_c, n_frames_total, weight

    def get_descr(self, obj, label, env_objects):
        if label == 1:
            t = np.random.randint(0, len(self.descriptions[obj]))
            descr, descr_enc = self.descriptions[obj][t]
        else:
            # select an alternate language
            tt = np.random.random()
            alt_obj = env_objects[1:]
            if len(alt_obj) == 0:
                alt_obj = list(range(0,obj)) + list(range(obj+1,self.N_OBJ))
            obj_ = np.random.choice(alt_obj)
            t = np.random.randint(0, len(self.descriptions[obj_])-1)
            descr, descr_enc = self.descriptions[obj_][t]
        return descr, descr_enc

    def __getitem__(self, index):
        if index >= len(self) // 2:
            label = 0
            index -= len(self) // 2
        else:
            label = 1

        obj = index // (self.repeat * self.N_ENV)
        env = self.video_ids[index % self.N_ENV]
        env_objects = self.load_env_objects(obj, env)

        frames_r, frames_l, frames_c, n_frames_total, weight = self.load_frames(obj, env)
        descr, descr_enc = self.get_descr(obj, label, env_objects)
        label = 2*label - 1
        return frames_r, frames_l, frames_c, descr, descr_enc, len(frames_r), len(descr_enc), label, obj, env, weight

