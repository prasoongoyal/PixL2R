from data import Data, PadBatch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.stats import spearmanr


VOCAB_SIZE = 264

def lstm_helper(sequences, lengths, lstm):
    if len(sequences) == 1:
        output, (hidden, _) = lstm(sequences)
        return output, hidden[-1]

    ordered_len, ordered_idx = lengths.sort(0, descending=True)
    ordered_sequences = sequences[ordered_idx]
    # remove zero lengths
    try:
        nonzero = list(ordered_len).index(0)
    except ValueError:
        nonzero = len(ordered_len)

    sequences_packed = pack_padded_sequence(
        ordered_sequences[:nonzero], ordered_len[:nonzero],
        batch_first=True)
    output_nonzero, (hidden_nonzero, _) = lstm(sequences_packed)
    output_nonzero = pad_packed_sequence(output_nonzero, batch_first=True)[0]
    max_len = sequences.shape[1]
    max_len_true = output_nonzero.shape[1]
    output = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
    output_final = torch.zeros(
        len(sequences), max_len, output_nonzero.shape[-1])
    output[:nonzero, :max_len_true, :] = output_nonzero
    hidden = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden_final = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden[:nonzero, :] = hidden_nonzero[-1]
    output_final[ordered_idx] = output
    hidden_final[ordered_idx] = hidden
    return output_final.cuda(), hidden_final.cuda()

class ImgEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4*4*args.n_channels, args.img_enc_size),
            nn.Linear(args.img_enc_size, args.img_enc_size),
        )

    def forward(self, x):
        return self.encoder(x)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_enc_r = ImgEnc(args)
        self.img_enc_l = ImgEnc(args)
        self.img_enc_c = ImgEnc(args)

        # trajectory encoder
        self.traj_encoder = nn.LSTM(
            3 * args.img_enc_size, 
            args.img_enc_size, 
            batch_first=True, 
            num_layers=args.num_layers)

        # language encoder
        self.embedding = nn.Embedding(VOCAB_SIZE, args.lang_enc_size)
        self.descr_encoder = nn.LSTM(args.lang_enc_size, args.lang_enc_size, batch_first=True, num_layers=args.num_layers)

        # linear layers
        self.linear1 = nn.Linear(args.img_enc_size + args.lang_enc_size, args.classifier_size)
        self.linear2 = nn.Linear(args.classifier_size, 1)


    def forward(self, traj_r, traj_l, traj_c, lang, traj_len, lang_len):
        traj_r_enc = self.img_enc_r(traj_r.view(-1, *traj_r.shape[-3:]))
        traj_r_enc = traj_r_enc.view(*traj_r.shape[:2], -1)
        traj_l_enc = self.img_enc_l(traj_l.view(-1, *traj_l.shape[-3:]))
        traj_l_enc = traj_l_enc.view(*traj_l.shape[:2], -1)
        traj_c_enc = self.img_enc_c(traj_c.view(-1, *traj_c.shape[-3:]))
        traj_c_enc = traj_c_enc.view(*traj_c.shape[:2], -1)

        traj_enc = torch.cat([traj_r_enc, traj_l_enc, traj_c_enc], dim=-1)
        _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        lang_emb = self.embedding(lang)
        _, lang_enc = lstm_helper(lang_emb, lang_len, self.descr_encoder)

        traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        pred = F.relu(self.linear1(traj_lang))
        pred = self.linear2(pred)
        return pred, lang_emb

class Predict:
    def __init__(self, model_file, lr, n_updates):
        ckpt = torch.load(model_file)
        self.args = ckpt['args']
        self.model = Model(self.args).cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.)
        self.n_updates = n_updates

    def predict_scores(self, traj, lang, traj_len, lang_len):
        self.model.eval()
        scores = np.zeros((len(traj), len(traj)))
        for start in range(len(traj)):
            for end in range(start+1, len(traj)):
                traj_sampled = traj[start:end, :, :, :]
                traj_sampled = np.array(traj_sampled)
                traj_sampled = torch.from_numpy(traj_sampled)
                traj_sampled = traj_sampled.cuda().float()
                traj_sampled = torch.transpose(traj_sampled, 2, 3)
                traj_sampled = torch.transpose(traj_sampled, 1, 2)
                lang = lang.cuda().long()
                traj_len = torch.Tensor([end-start])
                lang_len = torch.Tensor(lang_len)
                prob = self.model(torch.unsqueeze(traj_sampled, 0), torch.unsqueeze(lang, 0), traj_len, lang_len)
                prob_norm = torch.softmax(prob, dim=-1).data.cpu().numpy()
                scores[start, end] = (prob_norm[0, 1] - prob_norm[0, 0])
        return scores

    def predict_test(self, traj_r, traj_l, traj_c, lang, traj_len, lang_len):
        self.model.train()
        traj_r_sampled = traj_r
        traj_r_sampled = np.array(traj_r_sampled)
        traj_r_sampled = torch.from_numpy(traj_r_sampled)
        traj_r_sampled = traj_r_sampled.cuda().float()
        traj_r_sampled = torch.transpose(traj_r_sampled, 3, 4)
        traj_r_sampled = torch.transpose(traj_r_sampled, 2, 3)
        traj_l_sampled = traj_l
        traj_l_sampled = np.array(traj_l_sampled)
        traj_l_sampled = torch.from_numpy(traj_l_sampled)
        traj_l_sampled = traj_l_sampled.cuda().float()
        traj_l_sampled = torch.transpose(traj_l_sampled, 3, 4)
        traj_l_sampled = torch.transpose(traj_l_sampled, 2, 3)
        traj_c_sampled = traj_c
        traj_c_sampled = np.array(traj_c_sampled)
        traj_c_sampled = torch.from_numpy(traj_c_sampled)
        traj_c_sampled = traj_c_sampled.cuda().float()
        traj_c_sampled = torch.transpose(traj_c_sampled, 3, 4)
        traj_c_sampled = torch.transpose(traj_c_sampled, 2, 3)
        lang = lang.cuda().long()
        traj_len = torch.Tensor(traj_len)
        lang_len = torch.Tensor(lang_len)
        prob, lang_emb = self.model(traj_r_sampled, traj_l_sampled, traj_c_sampled, lang, traj_len, lang_len)
        return prob, lang_emb

    def predict(self, traj_r, traj_l, traj_c, lang):
        self.model.eval()
        with torch.no_grad():
            traj_r_sampled = traj_r[::-1][::10][::-1]
            traj_r_sampled = np.array(traj_r_sampled)
            traj_r_sampled = torch.from_numpy(traj_r_sampled)
            traj_r_sampled = traj_r_sampled.cuda().float()
            traj_r_sampled = torch.transpose(traj_r_sampled, 2, 3)
            traj_r_sampled = torch.transpose(traj_r_sampled, 1, 2)
            traj_l_sampled = traj_l[::-1][::10][::-1]
            traj_l_sampled = np.array(traj_l_sampled)
            traj_l_sampled = torch.from_numpy(traj_l_sampled)
            traj_l_sampled = traj_l_sampled.cuda().float()
            traj_l_sampled = torch.transpose(traj_l_sampled, 2, 3)
            traj_l_sampled = torch.transpose(traj_l_sampled, 1, 2)
            traj_c_sampled = traj_c[::-1][::10][::-1]
            traj_c_sampled = np.array(traj_c_sampled)
            traj_c_sampled = torch.from_numpy(traj_c_sampled)
            traj_c_sampled = traj_c_sampled.cuda().float()
            traj_c_sampled = torch.transpose(traj_c_sampled, 2, 3)
            traj_c_sampled = torch.transpose(traj_c_sampled, 1, 2)
            lang = lang.cuda().long()
            traj_len = torch.Tensor([len(traj_r_sampled)])
            lang_len = torch.Tensor([len(lang)])
            prob, _ = self.model(
                torch.unsqueeze(traj_r_sampled, 0) / 255., 
                torch.unsqueeze(traj_l_sampled, 0) / 255.,
                torch.unsqueeze(traj_c_sampled, 0) / 255., 
                torch.unsqueeze(lang, 0), traj_len, lang_len)
        return prob

    def update(self, traj_r, traj_l, traj_c, lang, label):
        self.model.train()
        traj_len = min(150, len(traj_r))
        traj_r = torch.from_numpy(np.array(traj_r[:traj_len]))
        traj_r = torch.transpose(traj_r, 2, 3)
        traj_r = torch.transpose(traj_r, 1, 2)
        traj_l = torch.from_numpy(np.array(traj_l[:traj_len]))
        traj_l = torch.transpose(traj_l, 2, 3)
        traj_l = torch.transpose(traj_l, 1, 2)
        traj_c = torch.from_numpy(np.array(traj_c[:traj_len]))
        traj_c = torch.transpose(traj_c, 2, 3)
        traj_c = torch.transpose(traj_c, 1, 2)
        lang = lang.cuda().long()
        lang_len = torch.Tensor([len(lang)])
        label = torch.Tensor([2*label - 1]).cuda()
        for _ in range(self.n_updates):
            while True:
                selected = np.random.random(traj_len) > 0.9
                if np.sum(selected) > 0:
                    break
            traj_r_ = traj_r[selected].cuda().float()
            traj_l_ = traj_l[selected].cuda().float()
            traj_c_ = traj_c[selected].cuda().float()
            traj_len_ = torch.Tensor([len(traj_r_)])
            self.optimizer.zero_grad()
            prob = self.model(
                torch.unsqueeze(traj_r_, 0) / 255.,
                torch.unsqueeze(traj_l_, 0) / 255., 
                torch.unsqueeze(traj_c_, 0) / 255.,
                torch.unsqueeze(lang, 0),
                traj_len_, lang_len)[:, 0]
            loss = torch.nn.MSELoss()(prob, label)
            loss.backward()
            self.optimizer.step()

class Train:
    def __init__(self, args, train_data_loader, valid_data_loader):
        self.args = args
        self.model = Model(args).cuda()
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr)

    def run_batch(self, traj_r, traj_l, traj_c, lang, traj_len, 
        lang_len, labels, weights, is_train):
        if is_train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        traj_r = traj_r.cuda().float()
        traj_l = traj_l.cuda().float()
        traj_c = traj_c.cuda().float()
        lang = lang.cuda().long()
        labels = torch.Tensor(labels).cuda().long()
        weights = weights.cuda().float()
        pred, _ = self.model(traj_r, traj_l, traj_c, lang, traj_len, lang_len)
        prob = torch.tanh(pred[:, 0])
        loss = torch.nn.MSELoss()(prob, weights*labels)
        pred = prob
        
        if is_train:
            loss.backward()
            self.optimizer.step()

        return pred, loss.item()

    def run_epoch(self, data_loader, is_train):
        pred_all = []
        labels_all = []
        loss_all = []
        for frames_r, frames_l, frames_c, descr, descr_enc, \
            traj_len, descr_len, labels, _, _, weights in data_loader:
            pred, loss = self.run_batch(
                frames_r, frames_l, frames_c, descr_enc, traj_len, 
                descr_len, labels, weights, is_train)
            pred_all += pred.tolist()
            labels_all += (weights * labels).tolist()
            loss_all.append(loss)
        t, p = spearmanr(pred_all, labels_all)
        return np.round(np.mean(loss_all), 2), t, None

    def train_model(self):
        best_val_acc = 0.
        epoch = 1
        while True:
            valid_loss, valid_acc, valid_cm = self.run_epoch(
                self.valid_data_loader, is_train=False)
            train_loss, train_acc, train_cm = self.run_epoch(
                self.train_data_loader, is_train=True)
            print('Epoch: {}\tTL: {:.2f}\tTA: {:.2f}\tVL: {:.2f}\tVA: {:.2f}'.format(
                epoch, train_loss, 100. * train_acc, valid_loss, 100. * valid_acc))
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                if self.args.save_path:
                    state = {
                        'args': self.args, 
                        'epoch': epoch, 
                        'best_val_acc': best_val_acc, 
                        'state_dict': self.model.state_dict(), 
                        # 'optimizer': self.optimizer.state_dict()
                    }                
                    torch.save(state, self.args.save_path)
            if epoch == self.args.max_epochs:
                break
            epoch += 1

def main(args):
    train_data = Data(mode='train')
    valid_data = Data(mode='valid')
    print(len(train_data))
    print(len(valid_data))
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadBatch(),
        num_workers=16)
    valid_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadBatch(),
        num_workers=16)
    Train(args, train_data_loader, valid_data_loader).train_model()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n-channels', type=int, default=256)
    parser.add_argument('--img-enc-size', type=int, default=512)
    parser.add_argument('--lang-enc-size', type=int, default=512)
    parser.add_argument('--classifier-size', type=int, default=1024)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--logdir', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
