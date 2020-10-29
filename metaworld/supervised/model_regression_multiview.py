from data_multiview import Data, PadBatch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
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
    output_final = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
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
            # nn.BatchNorm2d(3), 
            nn.Conv2d(3, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.BatchNorm2d(args.n_channels), 
            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.BatchNorm2d(args.n_channels), 
            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4*4*args.n_channels, args.img_enc_size),
            # nn.Dropout(),
            nn.Linear(args.img_enc_size, args.img_enc_size),
        )

    def forward(self, x):
        return self.encoder(x)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        '''
        # image encoder
        self.img_enc = nn.Sequential(
            # nn.BatchNorm2d(3), 
            nn.Conv2d(3, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.BatchNorm2d(args.n_channels), 
            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.BatchNorm2d(args.n_channels), 
            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4*4*args.n_channels, args.img_enc_size),
            # nn.Dropout(),
            nn.Linear(args.img_enc_size, args.img_enc_size),
        )
        '''

        self.img_enc_right = ImgEnc(args)
        self.img_enc_left = ImgEnc(args)
        self.img_enc_center = ImgEnc(args)

        # trajectory encoder
        self.traj_encoder = nn.LSTM(3 * args.img_enc_size, args.img_enc_size, batch_first=True, num_layers=args.num_layers)

        # language encoder
        self.embedding = nn.Embedding(VOCAB_SIZE, args.lang_enc_size)
        self.descr_encoder = nn.LSTM(args.lang_enc_size, args.lang_enc_size, batch_first=True, num_layers=args.num_layers)

        # linear layers
        self.linear1 = nn.Linear(args.img_enc_size + args.lang_enc_size, args.classifier_size)
        self.linear2 = nn.Linear(args.classifier_size, 1)


    def forward(self, traj_right, traj_left, traj_center, lang, traj_len, lang_len):
        traj_right_enc = self.img_enc_right(traj_right.view(-1, *traj_right.shape[-3:]))
        traj_right_enc = traj_right_enc.view(*traj_right.shape[:2], -1)
        traj_left_enc = self.img_enc_left(traj_left.view(-1, *traj_left.shape[-3:]))
        traj_left_enc = traj_left_enc.view(*traj_left.shape[:2], -1)
        traj_center_enc = self.img_enc_center(traj_center.view(-1, *traj_center.shape[-3:]))
        traj_center_enc = traj_center_enc.view(*traj_center.shape[:2], -1)

        traj_enc = torch.cat([traj_right_enc, traj_left_enc, traj_center_enc], dim=-1)
        _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        lang_emb = self.embedding(lang)
        _, lang_enc = lstm_helper(lang_emb, lang_len, self.descr_encoder)

        traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        # traj_lang = F.dropout(traj_lang)
        pred = F.relu(self.linear1(traj_lang))
        # pred = F.dropout(pred)
        pred = self.linear2(pred)
        return pred, lang_emb

class Predict:
    def __init__(self, model_file, lr, n_updates):
        # from argparse import Namespace
        # args = Namespace(n_channels=64, img_enc_size=128, lang_enc_size=128, classifier_size=512, num_layers=2)
        # args = Namespace(n_channels=512, img_enc_size=512, lang_enc_size=512, classifier_size=1024, num_layers=1)
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

    def predict_test(self, traj_right, traj_left, traj_center, lang, traj_len, lang_len):
        self.model.train()
        traj_right_sampled = traj_right
        traj_right_sampled = np.array(traj_right_sampled)
        traj_right_sampled = torch.from_numpy(traj_right_sampled)
        traj_right_sampled = traj_right_sampled.cuda().float()
        traj_right_sampled = torch.transpose(traj_right_sampled, 3, 4)
        traj_right_sampled = torch.transpose(traj_right_sampled, 2, 3)
        traj_left_sampled = traj_left
        traj_left_sampled = np.array(traj_left_sampled)
        traj_left_sampled = torch.from_numpy(traj_left_sampled)
        traj_left_sampled = traj_left_sampled.cuda().float()
        traj_left_sampled = torch.transpose(traj_left_sampled, 3, 4)
        traj_left_sampled = torch.transpose(traj_left_sampled, 2, 3)
        traj_center_sampled = traj_center
        traj_center_sampled = np.array(traj_center_sampled)
        traj_center_sampled = torch.from_numpy(traj_center_sampled)
        traj_center_sampled = traj_center_sampled.cuda().float()
        traj_center_sampled = torch.transpose(traj_center_sampled, 3, 4)
        traj_center_sampled = torch.transpose(traj_center_sampled, 2, 3)
        lang = lang.cuda().long()
        traj_len = torch.Tensor(traj_len)
        lang_len = torch.Tensor(lang_len)
        prob, lang_emb = self.model(traj_right_sampled, traj_left_sampled, traj_center_sampled, lang, traj_len, lang_len)
        return prob, lang_emb

    def predict(self, traj_right, traj_left, traj_center, lang):
        self.model.eval()
        with torch.no_grad():
            traj_right_sampled = traj_right[::-1][::10][::-1]
            traj_right_sampled = np.array(traj_right_sampled)
            traj_right_sampled = torch.from_numpy(traj_right_sampled)
            traj_right_sampled = traj_right_sampled.cuda().float()
            traj_right_sampled = torch.transpose(traj_right_sampled, 2, 3)
            traj_right_sampled = torch.transpose(traj_right_sampled, 1, 2)
            traj_left_sampled = traj_left[::-1][::10][::-1]
            traj_left_sampled = np.array(traj_left_sampled)
            traj_left_sampled = torch.from_numpy(traj_left_sampled)
            traj_left_sampled = traj_left_sampled.cuda().float()
            traj_left_sampled = torch.transpose(traj_left_sampled, 2, 3)
            traj_left_sampled = torch.transpose(traj_left_sampled, 1, 2)
            traj_center_sampled = traj_center[::-1][::10][::-1]
            traj_center_sampled = np.array(traj_center_sampled)
            traj_center_sampled = torch.from_numpy(traj_center_sampled)
            traj_center_sampled = traj_center_sampled.cuda().float()
            traj_center_sampled = torch.transpose(traj_center_sampled, 2, 3)
            traj_center_sampled = torch.transpose(traj_center_sampled, 1, 2)
            lang = lang.cuda().long()
            traj_len = torch.Tensor([len(traj_right_sampled)])
            lang_len = torch.Tensor([len(lang)])
            prob = self.model(torch.unsqueeze(traj_right_sampled, 0) / 255., torch.unsqueeze(traj_left_sampled, 0) / 255., \
                torch.unsqueeze(traj_center_sampled, 0) / 255., torch.unsqueeze(lang, 0), traj_len, lang_len)
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
        params_img_enc = list(self.model.img_enc_right.parameters()) + list(self.model.img_enc_left.parameters()) + list(self.model.img_enc_center.parameters())
        params_lang_enc = list(self.model.embedding.parameters()) + list(self.model.descr_encoder.parameters())
        params_rest = list(filter(lambda kv: 'img_enc' not in kv[0] and 'embedding' not in kv[0] and 'descr_enc' not in kv[0], self.model.named_parameters()))
        params_rest = list(map(lambda x: x[1], params_rest))
        self.optimizer_img_enc = optim.Adam(
            # self.model.parameters(), 
            params_img_enc,
            lr=self.args.lr_img_enc, 
            weight_decay=self.args.weight_decay)
        self.optimizer_lang_enc = optim.Adam(
            params_lang_enc,
            # self.model.parameters(), 
            lr=self.args.lr_lang_enc, 
            weight_decay=self.args.weight_decay)
        self.optimizer_rest = optim.Adam(
            params_rest,
            # self.model.parameters(), 
            lr=self.args.lr_rest, 
            weight_decay=self.args.weight_decay)

        # tensorboard
        if args.logdir:
            self.writer = SummaryWriter(log_dir=args.logdir)
            self.global_step = 0

    def run_batch(self, traj_right, traj_left, traj_center, lang, traj_len, lang_len, labels, weights, is_train):
        if is_train:
            self.model.train()
            self.optimizer_img_enc.zero_grad()
            self.optimizer_lang_enc.zero_grad()
            self.optimizer_rest.zero_grad()
        else:
            self.model.eval()

        traj_right = traj_right.cuda().float()
        traj_left = traj_left.cuda().float()
        traj_center = traj_center.cuda().float()
        lang = lang.cuda().long()
        labels = torch.Tensor(labels).cuda().long()
        weights = weights.cuda().float()
        prob = torch.tanh(self.model(traj_right, traj_left, traj_center, lang, traj_len, lang_len))[:, 0]
        loss = torch.nn.MSELoss()(prob, weights*labels)
        '''
        loss = torch.nn.CrossEntropyLoss(reduction='none')(prob, labels)
        loss = torch.mean(weights * loss)
        pred = torch.argmax(prob, dim=-1)
        pred = torch.sign(prob)
        '''
        pred = prob
        
        if is_train:
            loss.backward()
            self.optimizer_img_enc.step()
            self.optimizer_lang_enc.step()
            self.optimizer_rest.step()
            # tensorboard
            if self.args.logdir:
                self.global_step += 1
                if self.global_step % 100 == 0:
                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.writer.add_histogram(tag, value.data.cpu().numpy(), self.global_step)
                        self.writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), self.global_step)

        return pred, loss.item()

    def run_epoch(self, data_loader, is_train):
        pred_all = []
        labels_all = []
        loss_all = []
        for frames_right, frames_left, frames_center, descr, descr_enc, traj_len, descr_len, labels, _, _, weights in data_loader:
            pred, loss = self.run_batch(frames_right, frames_left, frames_center, descr_enc, traj_len, descr_len, labels, weights, is_train)
            pred_all += pred.tolist()
            labels_all += (weights * labels).tolist()
            loss_all.append(loss)
        # correct = [1.0 if x == y else 0.0 for (x, y) in zip(pred_all, labels_all)]
        t, p = spearmanr(pred_all, labels_all)
        # conf_mat = confusion_matrix(labels_all, pred_all)
        return np.round(np.mean(loss_all), 2), t, None

    def train_model(self):
        best_val_acc = 0.
        epoch = 1
        while True:
            # self.valid_data_loader.dataset.set_thresh(min(0.5, epoch / 100.))
            # self.train_data_loader.dataset.set_thresh(min(0.5, epoch / 100.))
            valid_loss, valid_acc, valid_cm = self.run_epoch(self.valid_data_loader, is_train=False)
            train_loss, train_acc, train_cm = self.run_epoch(self.train_data_loader, is_train=True)
            # print(train_cm)
            # print(valid_cm)
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
    train_data = Data(mode='train', sampling=args.sampling, prefix=args.prefix)
    valid_data = Data(mode='valid', sampling=args.sampling, prefix=args.prefix)
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
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-channels', type=int, default=64)
    parser.add_argument('--img-enc-size', type=int, default=128)
    parser.add_argument('--lang-enc-size', type=int, default=128)
    parser.add_argument('--classifier-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-img-enc', type=float, default=1e-4)
    parser.add_argument('--lr-lang-enc', type=float, default=1e-4)
    parser.add_argument('--lr-rest', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--sampling', default='random')
    parser.add_argument('--prefix', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()   
    main(args)
