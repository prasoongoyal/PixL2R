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


VOCAB_SIZE = 264

def lstm_helper(sequences, lengths, lstm):
    if len(sequences) == 1:
        output, (hidden, _) = lstm(sequences)
        return output, hidden[0]

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
    output_final = output
    hidden_final = hidden
    return output_final.cuda(), hidden_final.cuda()

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # image encoder
        self.img_enc = nn.Sequential(
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

        # trajectory encoder
        self.traj_encoder = nn.LSTM(args.img_enc_size, args.img_enc_size, batch_first=True)

        # language encoder
        self.embedding = nn.Embedding(VOCAB_SIZE, args.lang_enc_size)
        self.descr_encoder = nn.LSTM(args.lang_enc_size, args.lang_enc_size, batch_first=True)

        # linear layers
        self.linear1 = nn.Linear(args.img_enc_size + args.lang_enc_size, args.classifier_size)
        self.linear2 = nn.Linear(args.classifier_size, 2)

    def forward(self, traj, lang, traj_len, lang_len):
        traj_enc = self.img_enc(traj.view(-1, *traj.shape[-3:]))
        traj_enc = traj_enc.view(*traj.shape[:2], -1)
        _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        lang_enc = self.embedding(lang)
        _, lang_enc = lstm_helper(lang_enc, lang_len, self.descr_encoder)

        traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        pred = F.relu(self.linear1(traj_lang))
        pred = self.linear2(pred)
        return pred

class Predict:
    def __init__(self, model_file):
        from argparse import Namespace
        args = Namespace(n_channels=64, img_enc_size=128, lang_enc_size=128, classifier_size=512)
        ckpt = torch.load(model_file)
        self.args = args
        self.model = Model(args).cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def predict_scores(self, traj, lang, traj_len, lang_len):
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

    def predict_test(self, traj, lang, traj_len, lang_len):
        traj_sampled = traj
        traj_sampled = np.array(traj_sampled)
        traj_sampled = torch.from_numpy(traj_sampled)
        traj_sampled = traj_sampled.cuda().float()
        traj_sampled = torch.transpose(traj_sampled, 3, 4)
        traj_sampled = torch.transpose(traj_sampled, 2, 3)
        lang = lang.cuda().long()
        traj_len = torch.Tensor(traj_len)
        lang_len = torch.Tensor(lang_len)
        prob = self.model(traj_sampled, lang, traj_len, lang_len)
        return prob

    def predict(self, traj, lang):
        with torch.no_grad():
            traj_sampled = traj[::-1][::5][::-1]
            traj_sampled = np.array(traj_sampled)
            traj_sampled = torch.from_numpy(traj_sampled)
            traj_sampled = traj_sampled.cuda().float()
            traj_sampled = torch.transpose(traj_sampled, 2, 3)
            traj_sampled = torch.transpose(traj_sampled, 1, 2)
            lang = lang.cuda().long()
            traj_len = torch.Tensor([len(traj_sampled)])
            lang_len = torch.Tensor([len(lang)])
            prob = self.model(torch.unsqueeze(traj_sampled, 0) / 255., torch.unsqueeze(lang, 0), traj_len, lang_len)
        return prob

class Train:
    def __init__(self, args, train_data_loader, valid_data_loader):
        self.args = args
        self.model = Model(args).cuda()
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        params_img_enc = list(self.model.img_enc.parameters())
        params_rest = list(filter(lambda kv: 'img_enc' not in kv[0], self.model.named_parameters()))
        self.optimizer_img_enc = optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr_img_enc, 
            weight_decay=self.args.weight_decay)
        self.optimizer_rest = optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr_rest, 
            weight_decay=self.args.weight_decay)

    def run_batch(self, traj, lang, traj_len, lang_len, labels, is_train):
        if is_train:
            self.model.train()
            self.optimizer_img_enc.zero_grad()
            self.optimizer_rest.zero_grad()
        else:
            self.model.eval()

        traj = traj.cuda().float()
        lang = lang.cuda().long()
        labels = torch.Tensor(labels).cuda().long()
        prob = self.model(traj, lang, traj_len, lang_len)
        loss = torch.nn.CrossEntropyLoss()(prob, labels)
        pred = torch.argmax(prob, dim=-1)

        if is_train:
            loss.backward()
            self.optimizer_img_enc.step()
            self.optimizer_rest.step()

        return pred, loss.item()

    def run_epoch(self, data_loader, is_train):
        pred_all = []
        labels_all = []
        loss_all = []
        for frames, descr, traj_len, descr_len, labels, _, _ in data_loader:
            pred, loss = self.run_batch(frames, descr, traj_len, descr_len, labels, is_train)
            pred_all += pred.tolist()
            labels_all += labels
            loss_all.append(loss)
        correct = [1.0 if x == y else 0.0 for (x, y) in zip(pred_all, labels_all)]
        conf_mat = confusion_matrix(labels_all, pred_all)
        return np.round(np.mean(loss_all), 2), np.mean(correct), conf_mat

    def train_model(self):
        best_val_acc = 0.
        epoch = 1
        while True:
            # self.valid_data_loader.dataset.set_thresh(min(0.5, epoch / 100.))
            # self.train_data_loader.dataset.set_thresh(min(0.5, epoch / 100.))
            valid_loss, valid_acc, valid_cm = self.run_epoch(self.valid_data_loader, is_train=False)
            train_loss, train_acc, train_cm = self.run_epoch(self.train_data_loader, is_train=True)
            print(train_cm)
            print(valid_cm)
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
        num_workers=32)
    valid_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadBatch(),
        num_workers=32)
    Train(args, train_data_loader, valid_data_loader).train_model()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-channels', type=int, default=64)
    parser.add_argument('--img-enc-size', type=int, default=128)
    parser.add_argument('--lang-enc-size', type=int, default=128)
    parser.add_argument('--classifier-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-img-enc', type=float, default=1e-5)
    parser.add_argument('--lr-rest', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sampling', default=None)
    parser.add_argument('--prefix', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()   
    main(args)
