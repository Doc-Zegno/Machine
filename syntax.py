import sys
import math
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from vectorizer import Vectorizer
from filler import ZeroFiller


# Transforms sentences from corpus.
# Returns forms only
class FormExtractor:
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        for sent in self.sents:
            forms = []
            for dep, rel, head in sent:
                forms.append(dep[0])
            yield forms


# Transforms sentences from corpus.
# Returns words only
class WordExtractor:
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        for sent in self.sents:
            words = []
            for dep, rel, head in sent:
                words.append(dep[1])
            yield words


# Transforms sentences from corpus.
# Returns tags only
class TagExtractor:
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        for sent in self.sents:
            tags = []
            for dep, rel, head in sent:
                tags.append(dep[2])
            yield tags


def graph_descending(index, probabilities, visited, mapping):
    to_be_visited = set()
    probs = probabilities[index]

    # Is there any candidates?
    if probs:
        maximum = probs[0][1]
        for i in range(len(probs)):
            prob = probs[i]
            if math.isclose(maximum, prob[1]):
                cur_index = prob[0]
                if cur_index not in visited and cur_index != index:
                    to_be_visited.add(cur_index)
                    visited.add(cur_index)
            else:
                break
        probabilities[index] = probs[i:]

        for cur_index in to_be_visited:
            mapping[cur_index - 1] = index
            graph_descending(cur_index, probabilities, visited, mapping)


def try_build_tree(root, heads, unvisited, outliers):
    visited = {root}

    # Repeatedly connect
    updated = True
    while updated:
        updated = False
        for node in unvisited.copy():
            if heads[node] - 1 in visited:
                updated = True
                visited.add(node)
                unvisited.remove(node)

    # Now 'unvisited' contains only unresolved references.
    # It's safe to merge it with 'outliers'
    outliers.update(unvisited)
    return root, visited, outliers


def try_expand_tree(index, probabilities, heads, visited, outliers):
    probs = probabilities[index]
    head = 0
    for i in range(1, len(probs)):
        if probs[i][0] - 1 in visited:
            head = probs[i][0]
            break

    visited = visited.copy()
    outliers = outliers.copy()
    visited.add(index)
    outliers.remove(index)

    for node in outliers.copy():
        if heads[node] - 1 in visited:
            visited.add(node)
            outliers.remove(node)

    return index, head, visited, outliers


class SyntaxParser(nn.Module):
    def __init__(self, num_words, word_size, num_tags, tag_size,
                 input_size, hidden_size, num_rels):
        super(SyntaxParser, self).__init__()

        self.hidden_size = hidden_size

        # self.word_embedding = nn.Embedding(num_words, word_size)
        # self.tag_embedding = nn.Embedding(num_tags, tag_size)

        self.unit = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.W = torch.zeros((hidden_size * 2, hidden_size * 2))
        self.root = torch.zeros(1, hidden_size * 2)
        self.ones = torch.ones(1, self.hidden_size * 2)

        self.linear = nn.Linear(4 * hidden_size, num_rels)

        if torch.cuda.is_available():
            # self.word_embedding = self.word_embedding.cuda()
            # self.tag_embedding = self.tag_embedding.cuda()
            self.unit = self.unit.cuda()
            self.W = self.W.cuda()
            self.root = self.root.cuda()
            self.ones = self.ones.cuda()
            self.linear = self.linear.cuda()

        self.W = nn.Parameter(self.W)
        self.root = nn.Parameter(self.root)

    def forward(self, features):
        # features = torch.cat((word_embedding, tag_embedding), dim=1)
        seq_len = features.size()[0]
        # features = features.view(seq_len, 1, -1)

        # Output is SEQ_LEN x 1 x HIDDEN_SIZE
        output, hidden = self.unit(features)

        # Adding ReLU
        # output = F.relu(output)

        # Build correlation matrix
        output = output.view(seq_len, -1)
        root = Variable(self.ones) * self.root
        H = torch.cat((root, output))
        matching = output.mm(self.W.mm(H.t()))

        log_probabilities = F.log_softmax(matching, dim=1)
        probabilities = torch.exp(log_probabilities)

        # We can't retrieve hidden state for the head node
        # since it requires calculating argmax of probabilities for each word
        # which is not differentiable and thus can't be used during back propagation
        soft_hidden = probabilities.mm(H)
        paired_hidden = torch.cat((output, soft_hidden), dim=1)
        rels = self.linear(paired_hidden)
        rels = F.log_softmax(rels, dim=1)

        return log_probabilities, rels


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("too few arguments: path to train and test corpora expected")

    file_path = sys.argv[1]
    sents = []
    sent = []
    maximum = 0

    word_set = set()
    tag_set = set()
    rel_set = set()

    for line in open(file_path):
        # Skip blank lines
        if len(line) <= 1:
            continue

        # Skip comments
        if line[0] == '#':
            continue

        # Tokenize and get rid of ellipsis enhancement
        line = line.split()
        if line[0].find(".") != -1:
            continue

        # Reset sublist
        if line[0] == "1":
            if sent:
                sents.append(sent)
            sent = []

        head = int(line[6])
        if head > maximum:
            maximum = head

        tag = line[4]
        word = line[1].lower()
        rel = line[7]
        word_set.add(word)
        tag_set.add(tag)
        rel_set.add(rel)

        triple = ((word, tag), rel, head)
        sent.append(triple)

    sents.append(sent)
    print(tag_set)
    print(rel_set)

    word_to_ix = {w: i for i, w in enumerate(word_set)}
    tag_to_ix = {t: i for i, t in enumerate(tag_set)}
    rel_to_ix = {r: i for i, r in enumerate(rel_set)}
    ix_to_rel = {i: r for i, r in enumerate(rel_set)}
    num_words = len(word_set)
    num_tags = len(tag_set)
    num_rels = len(rel_set)

    ROOT_TAG = "root"

    WORD_SIZE = 100
    TAG_SIZE = 30
    HIDDEN_SIZE = 100
    NUM_EPOCHS = 3

    word_vectorizer = Vectorizer(WordExtractor(sents), None, "parser_word", WORD_SIZE,
                                 filler=ZeroFiller(WORD_SIZE), ce_enabled=False, tf_enabled=False)
    tag_vectorizer = Vectorizer(TagExtractor(sents), None, "parser_pos", TAG_SIZE,
                                filler=ZeroFiller(TAG_SIZE), ce_enabled=False, tf_enabled=False)

    parser = SyntaxParser(num_words, WORD_SIZE, num_tags, TAG_SIZE, WORD_SIZE + TAG_SIZE, HIDDEN_SIZE, num_rels)
    optimizer = optim.SGD(parser.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()
    start_point = time.time()

    # sents = sents[:1000]

    num_sents = len(sents)
    sents_per_interval = num_sents // 10

    is_cuda_available = torch.cuda.is_available()

    for epoch in range(NUM_EPOCHS):
        current_loss = 0
        next_dot = sents_per_interval
        current_sent = 0
        print("epoch #{}: [".format(epoch), end="", flush=True)

        for sent in sents:
            word_embeddings = []
            tag_embeddings = []
            targets_heads = []
            targets_rels = []

            for dep, rel, head in sent:
                word, tag = dep
                word_embeddings.append(word_vectorizer(word))
                tag_embeddings.append(tag_vectorizer(tag))
                targets_heads.append(head)
                targets_rels.append(rel_to_ix[rel])

            word_embeddings = Variable(torch.stack(word_embeddings))
            tag_embeddings = Variable(torch.stack(tag_embeddings))
            targets_heads = Variable(torch.LongTensor(targets_heads))
            targets_rels = Variable(torch.LongTensor(targets_rels))

            if is_cuda_available:
                word_embeddings = word_embeddings.cuda()
                tag_embeddings = tag_embeddings.cuda()
                targets_heads = targets_heads.cuda()
                targets_rels = targets_rels.cuda()

            # Prepare optimizer
            optimizer.zero_grad()
            out_head, out_rel = parser(word_embeddings, tag_embeddings)
            loss = loss_function(out_head, targets_heads)
            loss += loss_function(out_rel, targets_rels)
            current_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            # Emulate progress bar
            current_sent += 1
            if current_sent >= next_dot:
                next_dot += sents_per_interval
                print('💪', end="", flush=True)

        print("], average loss = {}".format(current_loss / len(sents)))

    print("time elapsed: {} s".format(int(time.time() - start_point)))
    torch.save(parser, "parser.pt")

    # Testing
    num_correct_heads = 0  # UAS
    num_correct_labeled_heads = 0  # LAS
    num_correct_modified_heads = 0
    num_correct_rels = 0
    num_correct_sents = 0  # UEM
    num_correct_labeled_sents = 0  # LEM
    num_correct_modified_sents = 0
    num_outliers = 0
    num_false_outliers = 0
    num_roots = 0
    num_words = 0
    sents_wo_roots = []

    file_path = sys.argv[2]
    sents = []
    sent = []
    for line in open(file_path):
        # Skip blank lines
        if len(line) <= 1:
            continue

        # Skip comments
        if line[0] == '#':
            continue

        # Tokenize and get rid of ellipsis enhancement
        line = line.split()
        if line[0].find(".") != -1:
            continue

        # Reset sublist
        if line[0] == "1":
            if sent:
                sents.append(sent)
            sent = []

        head = int(line[6])
        tag = line[4]
        word = line[1].lower()
        rel = line[7]

        triple = ((word, tag), rel, head)
        sent.append(triple)

    sents.append(sent)

    unknown_words = 0
    unknown_tags = 0
    unknown_rels = 0

    fout = open("result.conllu", "w")

    num_sents = len(sents)
    sents_per_interval = num_sents // 10
    next_dot = sents_per_interval
    current_sent = 0
    print("testing: [", end="", flush=True)

    word_vectorizer.reset_counters()
    tag_vectorizer.reset_counters()

    for sent in sents:
        word_embeddings = []
        tag_embeddings = []
        targets_heads = []
        targets_rels = []

        words = []
        tags = []

        for dep, rel, head in sent:
            word, tag = dep
            words.append(word)
            tags.append(tag)

            word_embeddings.append(word_vectorizer(word))
            tag_embeddings.append(tag_vectorizer(tag))

            targets_heads.append(head)
            try:
                targets_rels.append(rel_to_ix[rel])
            except KeyError:
                unknown_rels += 1
                targets_rels.append(0)

        word_embeddings = Variable(torch.stack(word_embeddings))
        tag_embeddings = Variable(torch.stack(tag_embeddings))

        if is_cuda_available:
            word_embeddings = word_embeddings.cuda()
            tag_embeddings = tag_embeddings.cuda()

        # Get and check output
        out_head, out_rel = parser(word_embeddings, tag_embeddings)
        num_words += len(sent)

        fout.write("#text = {}\n".format(" ".join(words)))

        # Find the most possibly root node.
        # Serious business
        # root = probabilities[0][0][0]
        # visited = {root}
        # mapping = [0 for i in range(len(sent))]
        # while len(visited) != len(sent):
        #     graph_descending(root, probabilities, visited, mapping)
        # print("yay")

        heads = [0 for i in range(len(sent))]
        rels = ["" for i in range(len(sent))]
        probabilities = [[] for i in range(len(sent))]

        is_sent_correct = True
        is_labeled_sent_correct = True
        for i in range(len(sent)):
            for j in range(len(sent) + 1):
                probabilities[i].append((j, out_head.data[i][j]))
            probabilities[i].sort(key=lambda pair: pair[1], reverse=True)

            maximum, indices = out_head[i].max(0)
            predicted = indices.data[0]
            expected = targets_heads[i]
            head = predicted
            heads[i] = head
            is_head_correct = expected == predicted
            if is_head_correct:
                num_correct_heads += 1
            else:
                is_sent_correct = False
                is_labeled_sent_correct = False

            maximum, indices = out_rel[i].max(0)
            predicted = indices.data[0]
            expected = targets_rels[i]
            rel = ix_to_rel[predicted]
            rels[i] = rel
            if expected == predicted:
                num_correct_rels += 1
                if is_head_correct:
                    num_correct_labeled_heads += 1
            else:
                is_labeled_sent_correct = False

        if is_sent_correct:
            num_correct_sents += 1
        if is_labeled_sent_correct:
            num_correct_labeled_sents += 1

        out_head = out_head.data
        # Trying to turn random graph into well-formed tree.
        # Step 1. Find a root

        # maximum = out_head[0][0] - 1.0
        # index = -1
        # outliers = set()
        # unvisited = set()
        # for i in range(len(sent)):
        #     if heads[i] == 0:
        #         if out_head[i][0] > maximum:
        #             if index >= 0:
        #                 outliers.add(index)
        #             maximum = out_head[i][0]
        #             index = i
        #         else:
        #             outliers.add(i)
        #     else:
        #         unvisited.add(i)
        # root = index
        # visited = {root}

        outliers = set()
        roots = set()
        unvisited = set()
        for i in range(len(sent)):
            if heads[i] == 0 or rels[i] == ROOT_TAG:
                roots.add(i)
                # if rels[i] == ROOT_TAG:
                #     roots.add(i)
                # else:
                #     outliers.add(i)
            else:
                unvisited.add(i)

        num_roots += len(roots)
        if len(roots) > 0:
            options = []
            for node in roots:
                tmp_outliers = outliers.copy()
                tmp_outliers.update(roots)
                tmp_outliers.remove(node)
                options.append(try_build_tree(node, heads, unvisited.copy(), tmp_outliers))
            options.sort(key=lambda t: len(t[2]))
            root, visited, outliers = options[0]
        else:
            sents_wo_roots.append(current_sent)
            maximum = out_head[0][0] - 1.0
            index = -1
            for i in range(len(sent)):
                if out_head[i][0] > maximum:
                    maximum = out_head[i][0]
                    index = i
            root = index
            # heads[root] = 0
            # rels[root] = ROOT_TAG
            if root in outliers:
                outliers.remove(root)
            if root in unvisited:
                unvisited.remove(root)
            root, visited, outliers = try_build_tree(root, heads, unvisited, outliers)
        heads[root] = 0
        rels[root] = ROOT_TAG
        # visited = {root}

        # Repeatedly connect
        # updated = True
        # while updated:
        #     updated = False
        #     for node in unvisited.copy():
        #         if heads[node] - 1 in visited:
        #             updated = True
        #             visited.add(node)
        #             unvisited.remove(node)

        # Now 'unvisited' contains only unresolved references.
        # It's safe to merge it with 'outliers'
        # outliers.update(unvisited)
        num_outliers += len(outliers)

        for node in outliers:
            if heads[node] == targets_heads[node]:
                num_false_outliers += 1

        # era = 1
        # while len(outliers) > 0:
        #     updated = True
        #     while updated:
        #         updated = False
        #         for node in outliers.copy():
        #             try:
        #                 head = probabilities[node][era][0] - 1
        #             except IndexError:
        #                 print("IndexError: node = {}, era = {}, list is".format(node, era))
        #                 print(probabilities[node])
        #                 print(outliers)
        #                 raise RuntimeError("ololo")
        #             if head in visited:
        #                 updated = True
        #                 heads[node] = head + 1
        #                 visited.add(node)
        #                 outliers.remove(node)
        #     era += 1

        # Try to expand the tree
        while len(outliers) > 0:
            options = []
            for node in outliers:
                options.append(try_expand_tree(node, probabilities, heads, visited, outliers))
            options.sort(key=lambda t: len(t[3]))
            index, head, visited, outliers = options[0]
            heads[index] = head

        is_modified_sent_correct = True
        for i in range(len(sent)):
            if heads[i] == targets_heads[i]:
                num_correct_modified_heads += 1
            else:
                is_modified_sent_correct = False
            fout.write("{}\t{}\t{}\t_\t{}\t_\t{}\t{}\t{}:{}\t_\n".format(
                i + 1, words[i], words[i], tags[i], heads[i], rels[i], heads[i], rels[i]))
        fout.write("\n")

        if is_modified_sent_correct:
            num_correct_modified_sents += 1

        # Emulate progress bar
        current_sent += 1
        if current_sent >= next_dot:
            next_dot += sents_per_interval
            print('💪', end="", flush=True)

    print("]")
    fout.close()

    print("correct heads (UAS): {} from {} ({:4.2f}%)".format(num_correct_heads, num_words,
                                                              num_correct_heads / num_words * 100.0))
    print("correct labeled heads (LAS): {} from {} ({:4.2f}%)".format(
        num_correct_labeled_heads, num_words, num_correct_labeled_heads / num_words * 100.0
    ))
    print("UAS after modification: {} from {} ({:4.2f}%)".format(
        num_correct_modified_heads, num_words, num_correct_modified_heads / num_words * 100.0
    ))
    print("correct rels: {} from {} ({:4.2f}%)".format(num_correct_rels, num_words,
                                                       num_correct_rels / num_words * 100.0))
    print("outliers ratio: {} from {} ({:4.2f}%)".format(
        num_outliers, num_words, num_outliers / num_words * 100.0
    ))
    print("false outliers: {} from {} ({:4.2f}%)".format(
        num_false_outliers, num_outliers, num_false_outliers / num_outliers * 100.0
    ))
    print("predicted roots per sentence: {:3.1f}".format(num_roots / num_sents))
    print("unlabeled exact matches: {} from {} ({:4.2f}%)".format(
        num_correct_sents, num_sents, num_correct_sents / num_sents * 100.0
    ))
    print("labeled exact matches: {} from {} ({:4.2f}%)".format(
        num_correct_labeled_sents, num_sents, num_correct_labeled_sents / num_sents * 100.0
    ))
    print("UEM after modification: {} from {} ({:4.2f}%)".format(
        num_correct_modified_sents, num_sents, num_correct_modified_sents / num_sents * 100.0
    ))
    print("unknown words: {} from {} ({:4.2f}%)".format(
        word_vectorizer.num_word_misses, num_words, word_vectorizer.num_word_misses / num_words * 100.0))
    print("unknown tags: {} from {} ({:4.2f}%)".format(
        tag_vectorizer.num_word_misses, num_words, tag_vectorizer.num_word_misses / num_words * 100.0
    ))
    print("\ttags: {} ({:4.2f}%)".format(unknown_tags, unknown_tags / num_words * 100.0))
    print("\trels: {} ({:4.2f}%)".format(unknown_rels, unknown_rels / num_words * 100.0))
    print("sentences w/o roots:")
    print(sents_wo_roots)
