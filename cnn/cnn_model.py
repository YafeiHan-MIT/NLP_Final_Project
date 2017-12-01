import sys
import time
import argparse
import cnn_model_utils as util
from cnn_model_utils import say
from evaluation import Evaluation
from prettytable import PrettyTable
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


def normalize_2d(x, eps=1e-8):
    # x is batch*d
    # l2 is batch*1
    l2 = x.norm(2, 1).view(x.size(0), 1)
    return x / (l2 + eps)


def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2, 2).view(x.size(0), x.size(1), 1)
    return x / (l2 + eps)


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, filter_width):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(input_size, hidden_size, filter_width)
        self.pool = nn.MaxPool1d(filter_width)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, xt, xb):
        # turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len)
        xt = xt.transpose(0, 1).transpose(1, 2)
        xb = xb.transpose(0, 1).transpose(1, 2)

        xt = F.tanh(self.conv(xt))
        xb = F.tanh(self.conv(xb))

        xt = F.tanh(self.pool(xt))
        xb = F.tanh(self.pool(xb))

        xt = normalize_3d(xt)
        xb = normalize_3d(xb)

        return normalize_2d(xt.mean(2) + xb.mean(2))


class Model:
    def __init__(self, args, embedding, batch_size, max_epoch=50):
        self.args = args
        self.embedding = embedding
        self.padding_id = embedding.vocab_id_map["<padding>"]
        self.batch_size = batch_size
        self.max_epoch = max_epoch

    def train(self, cnn, criterion, optimizer, train_batches, dev=None, test=None):
        result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                   ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
        unchanged = 0
        best_dev = -1
        dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
        test_MAP = test_MRR = test_P1 = test_P5 = 0
        for epoch in xrange(self.max_epoch):
            unchanged += 1
            if unchanged > 15: break

            N = len(train_batches)

            train_loss = 0.0
            train_cost = 0.0

            for i in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[i]
                # get embedding for every word for titles and bodies and then reshape
                xt = self.embedding.forward(idts.ravel())
                xt = xt.reshape((idts.shape[0], idts.shape[1], self.embedding.n_d))
                xb = self.embedding.forward(idbs.ravel())
                xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.embedding.n_d))

                titles = Variable(torch.from_numpy(xt)).float()
                bodies = Variable(torch.from_numpy(xb)).float()

                if args.cuda:
                    titles = titles.cuda()
                    bodies = bodies.cuda()

                optimizer.zero_grad()
                outputs = cnn(titles, bodies)

                idps_tensor = Variable(torch.from_numpy(idps.ravel()).long())
                if args.cuda:
                    idps_tensor = idps_tensor.cuda()
                xp = torch.index_select(outputs, 0, idps_tensor)
                # number of source vec * 22 * hidden layer size
                # 22 is from 1 source vec, 1 pos vec, 20 neg vec
                xp = xp.view(idps.shape[0], idps.shape[1], self.embedding.n_d)

                query_vec = xp[:, 0, :].contiguous()
                pos_scores = F.cosine_similarity(query_vec, xp[:, 1, :]).view(query_vec.size(0), 1)
                neg_scores = F.cosine_similarity(query_vec.view(xp.size(0), 1, xp.size(2)), xp[:, 2:, :], 2)
                # x_scores = torch.cat([pos_scores, neg_scores], 1)
                # y_targets = torch.zeros(x_scores.size(0)).type(torch.LongTensor)
                #
                # if args.cuda:
                #     y_targets = Variable(y_targets).cuda()
                # else:
                #     y_targets = Variable(y_targets)
                #
                # loss = criterion(x_scores, y_targets)
                loss = self.max_margin_loss(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()
                train_loss += loss
                train_cost += (loss + args.l2_reg * sum([x.norm(2) for x in cnn.parameters()]))

                if i % 10 == 0:
                    say("\r{}/{}".format(i, N))

                if i == N - 1:
                    if dev is not None:
                        dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(args, dev, cnn)
                    if test is not None:
                        test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(args, test, cnn)

                    if dev_MRR > best_dev:
                        unchanged = 0
                        best_dev = dev_MRR
                        result_table.add_row(
                            [epoch] +
                            ["%.2f" % x for x in [dev_MAP, dev_MRR, dev_P1, dev_P5] +
                             [test_MAP, test_MRR, test_P1, test_P5]]
                        )

                    say("\r\n\n")
                    say("Epoch {}\tcost={}\tloss={}\tMRR={},{}\n".format(
                        epoch,
                        train_cost.data / (i + 1),
                        train_loss.data / (i + 1),
                        dev_MRR,
                        best_dev
                    ))
                    say("{}".format(result_table))
                    say("\n")

    def evaluate(self, args, data, cnn):
        res = []
        for idts, idbs, labels in data:
            xt = self.embedding.forward(idts.ravel())
            xt = xt.reshape((idts.shape[0], idts.shape[1], self.embedding.n_d))
            xb = self.embedding.forward(idbs.ravel())
            xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.embedding.n_d))
            titles = Variable(torch.from_numpy(xt)).float()
            bodies = Variable(torch.from_numpy(xb)).float()
            if args.cuda:
                titles = titles.cuda()
                bodies = bodies.cuda()
            outputs = cnn(titles, bodies)
            pos = outputs[0].view(1, outputs[0].size(0))
            scores = torch.mm(pos, outputs[1:].transpose(1,0)).squeeze()
            if args.cuda:
                scores = scores.data.cpu().numpy()
            else:
                scores = scores.data.numpy()
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)
        e = Evaluation(res)
        MAP = e.MAP() * 100
        MRR = e.MRR() * 100
        P1 = e.Precision(1) * 100
        P5 = e.Precision(5) * 100
        return MAP, MRR, P1, P5

    def max_margin_loss(self, pos_scores, neg_scores, margin=0.1):
        neg_scores_max, index_max = neg_scores.max(
            dim=1)  # max over all 20 negative samples  # 1-d Tensor: num source queries
        diff = neg_scores_max - pos_scores + margin  # 1-d tensor  length = num of source queries
        # average loss over all source queries in a batch
        loss = ((diff > 0).float() * diff).mean()  ##
        self.loss = loss  # Variable for back-propagation
        return loss


def main(args):
    raw_corpus = util.read_corpus(args.corpus)
    embedding = util.create_embedding_layer(raw_corpus, util.load_embedding_iterator(args.embeddings))
    ids_corpus = util.map_corpus(raw_corpus, embedding)
    padding_id = embedding.vocab_id_map["<padding>"]

    dev, test = None, None
    if args.dev:
        dev = util.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = util.create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)
    if args.test:
        test = util.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = util.create_eval_batches(ids_corpus, test, padding_id, pad_left=not args.average)

    if args.train:
        train = util.read_annotations(args.train)
        train_batches = util.create_batches(ids_corpus, train, args.batch_size,
                                            padding_id, pad_left=not args.average)
        model = Model(args, embedding, args.batch_size)
        input_size = embedding.n_d
        hidden_size = 200
        output_size = hidden_size
        filter_width = 3
        cnn = CNN(input_size, hidden_size, output_size, filter_width)
        criterion = nn.MultiMarginLoss(p=1, margin=0.1, size_average=True)
        if args.cuda:
            cnn = cnn.cuda()
            criterion = criterion.cuda()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
        model.train(cnn, criterion, optimizer, train_batches, dev, test)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus",
                           type=str
                           )
    argparser.add_argument("--train",
                           type=str,
                           default=""
                           )
    argparser.add_argument("--test",
                           type=str,
                           default=""
                           )
    argparser.add_argument("--dev",
                           type=str,
                           default=""
                           )
    argparser.add_argument("--embeddings",
                           type=str,
                           default=""
                           )
    argparser.add_argument("--average",
                           type=int,
                           default=0
                           )
    argparser.add_argument("--batch_size",
                           type=int,
                           default=40
                           )
    argparser.add_argument("--cuda",
                           type=bool,
                           default=False)
    argparser.add_argument("--l2_reg",
                           type=float,
                           default=1e-5
                           )

    args = argparser.parse_args()
    print(args)
    main(args)
