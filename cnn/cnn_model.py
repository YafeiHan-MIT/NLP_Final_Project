import sys
import time
import argparse
import cnn_model_utils as util
from cnn_model_utils import say
from evaluation import Evaluation
from prettytable import PrettyTable
from torch.autograd import Variable, Function
import torch.nn as nn
import torch.nn.functional as F
import torch


def normalize_2d(x, eps=1e-8):
    # x is batch*d
    # l2 is batch*1
    l2 = x.norm(2, axis=1).dimshuffle((0, 'x'))
    return x / (l2 + eps)

def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2,axis=2).dimshuffle((0,1,'x'))
    return x/(l2+eps)


class LossFunction(Function):

    def forward(self, outputs):
        pass


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, filter_width):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(input_size, hidden_size, filter_width)
        self.pool = nn.AvgPool1d(filter_width)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, xt, xb):
        # turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len)
        xt = xt.transpose(0, 1).transpose(1, 2)
        xb = xb.transpose(0, 1).transpose(1, 2)

        xt = self.pool(F.tanh(self.conv(xt)))
        xb = self.pool(F.tanh(self.conv(xb)))

        xt = normalize_3d(xt)
        xb = normalize_3d(xb)

        print xt.data.size()
        print xb.data.size()

        return normalize_2d(xt + xb)


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
        start_time = 0
        for epoch in xrange(self.max_epoch):
            unchanged += 1
            if unchanged > 15: break

            start_time = time.time()

            N = len(train_batches)

            train_loss = 0.0
            train_cost = 0.0

            for i in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[i]
                xt = self.embedding.forward(idts.ravel())
                xt = xt.reshape((idts.shape[0], idts.shape[1], self.embedding.n_d))
                xb = self.embedding.forward(idbs.ravel())
                xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.embedding.n_d))

                titles = Variable(torch.from_numpy(xt)).float()
                bodies = Variable(torch.from_numpy(xb)).float()

                optimizer.zero_grad()
                outputs = cnn(titles, bodies)
                xp = outputs[idps.ravel()]
                xp = xp.reshape((idps.shape[0], idps.shape[1], self.embedding.n_d))

                loss = criterion(outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss

                if i % 10 == 0:
                    say("\r{}/{}".format(i, N))

                if i == N - 1:
                    if dev is not None:
                        dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(dev, eval_func)
                    if test is not None:
                        test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(test, eval_func)

                    if dev_MRR > best_dev:
                        unchanged = 0
                        best_dev = dev_MRR
                        result_table.add_row(
                            [epoch] +
                            ["%.2f" % x for x in [dev_MAP, dev_MRR, dev_P1, dev_P5] +
                             [test_MAP, test_MRR, test_P1, test_P5]]
                        )

                    say("\r\n\n")
                    say(("Epoch {}\tcost={:.3f}\tloss={:.3f}" \
                         + "\tMRR={:.2f},{:.2f}\t[{:.3f}m]\n").format(
                        epoch,
                        train_cost / (i + 1),
                        train_loss / (i + 1),
                        dev_MRR,
                        best_dev,
                        (time.time() - start_time) / 60.0
                    ))
                    say("{}".format(result_table))
                    say("\n")

    def evaluate(self, data, eval_func):
        res = []
        for idts, idbs, labels in data:
            scores = eval_func(idts, idbs)
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


def main(args):
    raw_corpus = util.read_corpus(args.corpus)
    embedding = util.create_embedding_layer(raw_corpus, util.load_embedding_iterator(args.embeddings))
    ids_corpus = util.map_corpus(raw_corpus, embedding)
    padding_id = embedding.vocab_id_map["<padding>"]

    # if args.dev:
    #     dev = util.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
    #     dev = util.create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)
    # if args.test:
    #     test = util.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
    #     test = util.create_eval_batches(ids_corpus, test, padding_id, pad_left=not args.average)

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
        optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
        model.train(cnn, criterion, optimizer, train_batches)

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

    args = argparser.parse_args()
    print args
    main(args)
