import sys
import gzip
import numpy as np
import random
from collections import Counter


class Embedding:
    def __init__(self, vocab, embs, oov="<unk>"):
        """
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)
        :param vocab: an iterator of string tokens; the layer will allocate an ID and a vector for each token in it
        :param oov: out-of-vocabulary token
        :param embs: an iterator of (word, vector) pairs
        """
        words = []
        vocab_id_map = {}
        emb_vec = []

        for word, vector in embs:
            assert word not in vocab_id_map, "Duplicate words in initial embeddings"
            vocab_id_map[word] = len(vocab_id_map)
            emb_vec.append(vector)
            words.append(word)

        say("{} pre-trained embeddings loaded.\n".format(len(emb_vec)))

        n_d = len(emb_vec[0])

        for word in vocab:
            if word not in vocab_id_map:
                vocab_id_map[word] = len(vocab_id_map)
                emb_vec.append(np.random.rand(n_d,) * (0.001 if word != oov else 0.0))
                words.append(word)

        self.vocab_id_map = vocab_id_map
        self.words = words
        self.emb_vec = emb_vec

        assert oov in self.vocab_id_map, "oov {} not in vocab".format(oov)
        self.oov_token = oov
        self.oov_id = self.vocab_id_map[oov]

        self.n_V = len(self.vocab_id_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, words = self.n_V, self.words
        return [words[i] if i < n_V else "<err>" for i in ids]

    def map_to_ids(self, words, filter_oov=False):
        """
        map the list of string tokens into a numpy array of integer IDs
        :param words: the list of string tokens
        :param filter_oov: whether to remove oov tokens in the returned array
        :return:
        """
        vocab_id_map = self.vocab_id_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x != oov_id
            return np.array(
                filter(not_oov, [vocab_id_map.get(x, oov_id) for x in words]),
                dtype="int32"
            )
        else:
            return np.array(
                [vocab_id_map.get(x, oov_id) for x in words],
                dtype="int32"
            )

    def forward(self, x):
        """
        Return the word embeddings given word IDs x
        :param x: an array of integer IDs
        :return a numpy matrix of word embeddings
        """
        matrix = [self.emb_vec[i] for i in x]
        return np.array(matrix)


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([float(x) for x in parts[1:]])
                yield word, vals


def create_embedding_layer(raw_corpus, embs, cut_off=2, unk="<unk>", padding="<padding>"):
    cnt = Counter(w for id, pair in raw_corpus.iteritems() \
                  for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = Embedding(
        vocab=[unk, padding],
        embs=embs
    )
    return embedding_layer


def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = [title, body]
    print "{} empty titles ignored.\n".format(empty_cnt)
    return raw_corpus


def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = []
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = []
            qlabels = []
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))

    return lst


def map_corpus(raw_corpus, embedding, max_len=100):
    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():
        item = (embedding.map_to_ids(pair[0]),
                embedding.map_to_ids(pair[1])[:max_len])
        ids_corpus[id] = item
    return ids_corpus


def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):
    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)
    cnt = 0
    pid2id = {}
    titles = []
    bodies = []
    triples = []
    batches = []
    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus: continue
                pid2id[id] = len(titles)
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)
        pid = pid2id[pid]
        pos = [pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id]
        neg = [pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id]
        triples += [[pid, x] + neg for x in pos]

        if cnt == batch_size or u == N - 1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            triples = create_hinge_batch(triples)
            batches.append((titles, bodies, triples))
            titles = []
            bodies = []
            triples = []
            pid2id = {}
            cnt = 0
    return batches


def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = []
    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        for id in [pid] + qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    return lst


def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        titles = np.column_stack([np.pad(x, (max_title_len - len(x), 0), 'constant',
                                         constant_values=padding_id) for x in titles])
        bodies = np.column_stack([np.pad(x, (max_body_len - len(x), 0), 'constant',
                                         constant_values=padding_id) for x in bodies])
    else:
        titles = np.column_stack([np.pad(x, (0, max_title_len - len(x)), 'constant',
                                         constant_values=padding_id) for x in titles])
        bodies = np.column_stack([np.pad(x, (0, max_body_len - len(x)), 'constant',
                                         constant_values=padding_id) for x in bodies])
    return titles, bodies


def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([np.pad(x, (0, max_len - len(x)), 'edge')
                         for x in triples]).astype('int32')
    return triples


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()
