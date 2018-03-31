import os
import sys
import time
import math

from scipy.stats import hmean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import nltk
import nltk.corpus as corpus

from vectorizer import Vectorizer
from ft_vectorizer import FastTextVectorizer
from tagger import Tagger
from syntax import SyntaxParser, try_build_tree, try_expand_tree
from filler import ZeroFiller
from score import TagScore, ChunkScore, ParserScore


# Encapsulation of endless training and testing scripts
class Model:
    def __init__(self, name, sents, vectorizer_words, vectorizer_forms, embedding_size,
                 tag_sents, tag_embedding_size, context_size,
                 lrs=(0.1, 0.1, 0.1), lr_decrease_factor=0.5, epochs_per_decrease=10):
        ######################################################################
        # Model's parameters.
        # 'sents' is a list of sentences of tuples ((form, word, tag), rel, head)
        self.name = name
        self.sents = sents
        self.embedding_size = embedding_size
        self.context_size = context_size

        ######################################################################
        # Load or create indices.
        # Common
        self.path_base = "internal"
        self.num_words = 0
        self.root_tag = "root"

        # CUDA flag
        self.is_cuda_available = torch.cuda.is_available()

        # For POS tags:
        self.tags = set()
        self.num_tags = 0
        self.tag2index = {}
        self.index2tag = {}

        # For chunk tags:
        self.chunks = set()
        self.num_chunks = 0
        self.chunk2index = {}
        self.index2chunk = {}

        # For relation tags:
        self.rels = set()
        self.num_rels = 0
        self.rel2index = {}
        self.index2rel = {}

        # Update database
        self.create_or_load_indices()
        if self.num_words == 0:
            self.num_words = self.get_num_words(self.sents)

        ######################################################################
        # Logic.
        # Learning rate controls
        self.lrs = lrs
        self.lr_decrease_factor = lr_decrease_factor
        self.epochs_per_decrease = epochs_per_decrease

        # Define machines
        self.vectorizer = Vectorizer(vectorizer_words, vectorizer_forms, name,
                                     embedding_size, filler=ZeroFiller(embedding_size),
                                     ce_enabled=True)

        # self.vectorizer = FastTextVectorizer(name, embedding_size * 2, "ft_sg_syntagrus.bin")

        self.tag_vectorizer = Vectorizer(tag_sents, None, name + "_pos",
                                         tag_embedding_size, filler=ZeroFiller(tag_embedding_size),
                                         ce_enabled=False, tf_enabled=False)

        # Tags embeddings (H).
        # Chunker will get linear combination as an input:
        #    I = H^T * p
        #    p - probabilities vector
        self.tag_embeddings = []
        for i in range(self.num_tags):
            tag = self.index2tag[i].lower()
            self.tag_embeddings.append(self.tag_vectorizer(tag, tag))
        self.tag_embeddings = torch.stack(self.tag_embeddings)
        if self.is_cuda_available:
            self.tag_embeddings = self.tag_embeddings.cuda()

        # Vector size is 1 (TF) + 100 (Word embedding) + 100 (Char grams embedding)
        self.vector_size = self.vectorizer.get_vector_size()

        self.tag_size = self.tag_vectorizer.get_vector_size()

        # Chunk size.
        # Benchmark is 200 (POS hidden) + 201 (embedding) + NUM_TAGS (probabilities)
        self.chunk_size = 2 * embedding_size + self.vector_size + self.tag_size

        # Parse size -- input size for parser.
        # When chunking is not available, parse size is equal to chunk size
        self.parse_size = self.chunk_size

        self.log("tagger input size: {}".format(self.vector_size))
        self.log("chunker input size: {}".format(self.chunk_size))
        self.log("parser input size: {}".format(self.parse_size))

        self.tagger = Tagger(self.vector_size, self.num_tags, "GRU", embedding_size)
        # self.chunker = Tagger(self.chunk_size, self.num_chunks, "LSTM", embedding_size)
        self.parser = SyntaxParser(0, 0, 0, 0, self.parse_size, embedding_size, self.num_rels)

        self.is_tagger_trained = False
        # self.is_chunker_trained = False
        self.is_parser_trained = False

        self.tagger_name = "pos tagging"
        # self.chunker_name = "chunking"
        self.parser_name = "parsing"

        # Try to load from file
        self.tagger_path = "{}/model_pos_{}.pt".format(self.path_base, self.name)
        # self.chunker_path = "{}/model_chunk_{}.pt".format(self.path_base, self.name)
        self.parser_path = "{}/model_parse_{}.pt".format(self.path_base, self.name)

        if os.path.exists(self.tagger_path):
            self.log("Loading POS tagger")
            self.tagger = torch.load(self.tagger_path)
            self.tagger.unit.flatten_parameters()
            self.is_tagger_trained = True
            self.log("Done")

        # if os.path.exists(self.chunker_path):
        #     self.log("Loading chunker")
        #     self.chunker = torch.load(self.chunker_path)
        #     self.chunker.unit.flatten_parameters()
        #     self.is_chunker_trained = True
        #     self.log("Done")

        if os.path.exists(self.parser_path):
            self.log("Loading parser")
            self.parser = torch.load(self.parser_path)
            self.parser.unit.flatten_parameters()
            self.is_parser_trained = True
            self.log("Done")

    ##########################################################################
    def train(self, sents, num_epochs, machines):
        ######################################################################
        # Define optimizers
        tag_optimizer = optim.SGD(self.tagger.parameters(), lr=self.lrs[0])
        # chunk_optimizer = optim.SGD(self.chunker.parameters(), lr=self.lrs[1])
        chunk_optimizer = None

        # Parameters for both machines
        params = list(self.tagger.parameters()) + list(self.parser.parameters())
        parse_optimizer = optim.SGD(params, lr=self.lrs[2])
        tag_loss_function = nn.NLLLoss()
        chunk_loss_function = nn.NLLLoss()
        parse_loss_function = nn.NLLLoss()

        ######################################################################
        # Run loop
        start_time = time.time()
        for epoch in range(num_epochs):
            print("epoch #{}: ".format(epoch), end="", flush=True)
            optimizers = [tag_optimizer, chunk_optimizer, parse_optimizer]
            self.loop(sents, optimizers, [tag_loss_function, chunk_loss_function, parse_loss_function],
                      [None, None, None])
            self.decrease_lr(optimizers, epoch + 1, self.lr_decrease_factor,
                             self.epochs_per_decrease)
        print("elapsed: {} s".format(int(time.time() - start_time)))

        # Out misses
        # self.print_vectorizer_misses()

        # Save model
        torch.save(self.tagger, self.tagger_path)
        # torch.save(self.chunker, self.chunker_path)
        torch.save(self.parser, self.parser_path)

        self.log("Done")

    ##########################################################################
    def test(self, sents):
        # Collect statistics
        tag_score = TagScore(self.tags)
        # chunk_score = ChunkScore(self.chunks)
        chunk_score = None
        parse_score = ParserScore()

        num_correct_tags = 0
        num_correct_chunks = 0
        num_words = 0

        start_time = time.time()
        self.loop(sents, [None, None, None], [nn.NLLLoss(), nn.NLLLoss(), nn.NLLLoss()],
                  [tag_score, chunk_score, parse_score])
        print("elapsed: {} s".format(int(time.time() - start_time)))

        # Out statistics
        print("POS Tagging:")
        f1_s = []
        has_zero = False
        for tag in sorted(self.tags):
            stat = tag_score.stats[tag]

            if stat.num_gold_predicted == 0 or stat.num_gold == 0 or stat.num_predicted == 0:
                print("\tskipped: {:>5} ({} items)".format(tag, stat.num_gold))
                continue

            num_words += stat.num_gold
            num_correct_tags += stat.num_gold_predicted

            precision = stat.num_gold_predicted / max(stat.num_predicted, 1.0)
            recall = stat.num_gold_predicted / max(stat.num_gold, 1.0)

            f1 = 0.0
            if math.isclose(precision, 0.0) or math.isclose(recall, 0.0):
                has_zero = True
            else:
                f1 = hmean([precision, recall])

            f1_s.append(f1)

            print("\t{:>5}: P = {:4.2f}%, R = {:4.2f}%, F1 = {:4.2f}% ({} items)".format(
                tag, precision * 100, recall * 100, f1 * 100, stat.num_gold))

            # ratio = 0
            # if stat[1] != 0:
            #     ratio = stat[0] / stat[1]
            # ratio *= 100
            #
            # print("\t{:>4}: {:4} / {:4} = {:4.2f}%".format(tag, stat[0], stat[1], ratio))

        # print("Chunking:")
        # for chunk in sorted(self.chunks):
        #     stat = chunk_score.stats[chunk]
        #     num_correct_chunks += stat[0]
        #
        #     ratio = 0
        #     if stat[1] != 0:
        #         ratio = stat[0] / stat[1]
        #     ratio *= 100
        #
        #     print("\t{:>4}: {:4} / {:4} = {:4.2f}%".format(chunk, stat[0], stat[1], ratio))

        # self.print_vectorizer_misses()

        # POS aggregated
        print("Total words:", num_words)
        print("Correct POS tags:", num_correct_tags, "({:4.2f}%)".format(
            num_correct_tags / num_words * 100.0))

        average_f1 = 0.0
        if not has_zero:
            average_f1 = hmean(f1_s)
        print("Average F1 = {:4.2f}%".format(average_f1 * 100))

        # Chunks aggregated
        # print("Correct chunk tags:", num_correct_chunks, "({:4.2f}%)".format(
        #     num_correct_chunks / num_words * 100.0))
        # precision = chunk_score.num_retrieved_relevant / chunk_score.num_retrieved
        # recall = chunk_score.num_retrieved_relevant / chunk_score.num_relevant
        # f1_score = hmean([precision, recall])
        # print("\tPrecision: {:f}".format(precision))
        # print("\tRecall: {:f}".format(recall))
        # print("\tF1 score: {:f}".format(f1_score))

        print("Parsing:")
        print("\tUAS: {} from {} ({:4.2f}%)".format(parse_score.num_unlabeled_arcs, num_words,
                                                    parse_score.num_unlabeled_arcs / num_words * 100.0))
        print("\tLAS: {} from {} ({:4.2f}%)".format(
            parse_score.num_labeled_arcs, num_words,
            parse_score.num_labeled_arcs / num_words * 100.0
        ))
        print("\tMUAS: {} from {} ({:4.2f}%)".format(
            parse_score.num_modified_unlabeled_arcs, num_words,
            parse_score.num_modified_unlabeled_arcs / num_words * 100.0
        ))
        print("\tCRel: {} from {} ({:4.2f}%)".format(
            parse_score.num_labels, num_words,
            parse_score.num_labels / num_words * 100.0))
        print("\tUEM: {} from {} ({:4.2f}%)".format(
            parse_score.num_unlabeled_trees, len(sents),
            parse_score.num_unlabeled_trees / len(sents) * 100.0
        ))
        print("\tLEM: {} from {} ({:4.2f}%)".format(
            parse_score.num_labeled_trees, len(sents),
            parse_score.num_labeled_trees / len(sents) * 100.0
        ))
        print("\tMUEM: {} from {} ({:4.2f}%)".format(
            parse_score.num_modified_unlabeled_trees, len(sents),
            parse_score.num_modified_unlabeled_trees / len(sents) * 100.0
        ))

        self.log("Done")

    ##########################################################################
    # Lose control and decrease the pace
    # Warped and bewitched
    # It's time to erase
    @staticmethod
    def decrease_lr(optimizers, epoch, factor, epoch_interval):
        if epoch % epoch_interval != 0:
            return

        print("lr is multiplied by {:f}".format(factor))

        for optimizer in optimizers:
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= factor

    ##########################################################################
    # Used both by 'train' and 'test'.
    # The only difference is that optimizers mustn't be provided
    # during testing
    def loop(self, sents, optimizers, loss_functions, scores):
        tag_optimizer = optimizers[0]
        chunk_optimizer = optimizers[1]
        parse_optimizer = optimizers[2]

        tag_loss_function = loss_functions[0]
        chunk_loss_function = loss_functions[1]
        parse_loss_function = loss_functions[2]

        tag_scores = scores[0]
        chunk_scores = scores[1]
        parse_scores = scores[2]

        # Total number of words
        num_words = self.get_num_words(sents)
        words_per_interval = num_words // 10

        # Average losses
        tag_loss = 0
        chunk_loss = 0
        parse_loss = 0

        # Reset vectorizer
        self.vectorizer.reset_counters()

        # Reset progress bar
        print("progress [", end="", flush=True)
        current_word = 0
        next_interval = words_per_interval

        # Dump
        fout = open("result.conllu", "w")

        for sent in sents:
            # Reset optimizers and machines
            # if tag_optimizer is not None:
            #     tag_optimizer.zero_grad()
            # if chunk_optimizer is not None:
            #     chunk_optimizer.zero_grad()
            if parse_optimizer is not None:
                parse_optimizer.zero_grad()
            self.tagger.reset()
            # self.chunker.reset()

            ##############################################################
            # POS Tagger.
            # Prepare input for tagger and targets for chunker
            sequence = []
            tag_targets = []
            # chunk_targets = []
            parse_head_targets = []
            parse_rel_targets = []
            for dep, rel, head in sent:
                form, word, tag = dep
                sequence.append(self.vectorizer(form, word))
                tag_targets.append(self.tag2index[tag])
                # chunk_targets.append(self.chunk2index[chunk])
                parse_head_targets.append(head)
                parse_rel_targets.append(self.rel2index[rel])
            sequence = Variable(torch.stack(sequence, dim=0))
            tag_targets = Variable(torch.LongTensor(tag_targets))
            # chunk_targets = Variable(torch.LongTensor(chunk_targets))
            parse_head_targets = Variable(torch.LongTensor(parse_head_targets))
            parse_rel_targets = Variable(torch.LongTensor(parse_rel_targets))

            if self.is_cuda_available:
                tag_targets = tag_targets.cuda()
                # chunk_targets = chunk_targets.cuda()
                parse_head_targets = parse_head_targets.cuda()
                parse_rel_targets = parse_rel_targets.cuda()

            # Optimize tagger
            tag_output = self.tagger(sequence.view((len(sent), 1, -1)))
            current_loss = tag_loss_function(tag_output, tag_targets)
            tag_loss += current_loss.data[0]
            # if tag_optimizer is not None:
            #     current_loss.backward()
            #     tag_optimizer.step()

            ##############################################################
            # Chunking.
            # Prepare input for chunker
            sequence = Variable(sequence.data)
            probabilities = torch.exp(tag_output)
            probabilities = probabilities.mm(Variable(self.tag_embeddings))
            tagger_hidden = self.tagger.last_output.view((len(sent), -1))
            sequence = torch.cat((tagger_hidden, sequence, probabilities), dim=1)
            # sequence = Variable(torch.cat((sequence, probabilities), dim=1))
            # sequence = Variable(sequence)

            # Optimize chunker
            # chunk_output = self.chunker(sequence.view((len(sent), 1, -1)))
            # current_loss = chunk_loss_function(chunk_output, chunk_targets)
            # chunk_loss += current_loss.data[0]
            # if chunk_optimizer is not None:
            #     current_loss.backward()
            #     chunk_optimizer.step()

            # Optimize parser
            parse_output_heads, parse_output_rels = self.parser(sequence.view(len(sent), 1, -1))
            current_parser_loss = parse_loss_function(parse_output_heads, parse_head_targets)
            current_parser_loss += parse_loss_function(parse_output_rels, parse_rel_targets)
            parse_loss += current_parser_loss.data[0]

            current_loss += current_parser_loss

            if parse_optimizer is not None:
                current_loss.backward()
                parse_optimizer.step()

            ##################################################################
            # Collect stats if necessary
            actual_chunks = []
            is_sent_correct = True
            is_labeled_sent_correct = True
            heads = [0 for i in range(len(sent))]
            rels = ["" for i in range(len(sent))]
            probabilities = [[] for i in range(len(sent))]
            forms = []
            words = []
            tags = []

            if tag_scores is not None or chunk_scores is not None or parse_scores is not None:
                # Output is SEQ_LEN x NUM_TAGS
                for i in range(len(sent)):
                    forms.append(sent[i][0][0])
                    words.append(sent[i][0][1])

                    if tag_scores is not None:
                        maximum, indices = tag_output[i].max(0)
                        predicted = indices.data[0]
                        expected = tag_targets.data[i]

                        tag = sent[i][0][2]
                        stat = tag_scores.stats[tag]
                        stat.num_gold += 1

                        predicted_tag = self.index2tag[predicted]
                        tags.append(predicted_tag)

                        if predicted == expected:
                            stat.num_gold_predicted += 1

                        tag_scores.stats[predicted_tag].num_predicted += 1

                    # if chunk_scores is not None:
                    #     maximum, indices = chunk_output[i].max(0)
                    #     predicted = indices.data[0]
                    #     expected = chunk_targets.data[i]
                    #
                    #     chunk = sent[i][2]
                    #     stat = chunk_scores.stats[chunk]
                    #     stat[1] += 1
                    #
                    #     if chunk[0] == "B":
                    #         chunk_scores.num_relevant += 1
                    #
                    #     actual_chunk = self.index2chunk[predicted]
                    #     actual_chunks.append(actual_chunk)
                    #     if actual_chunk[0] == "B":
                    #         chunk_scores.num_retrieved += 1
                    #
                    #     if predicted == expected:
                    #         stat[0] += 1

                    if parse_scores is not None:
                        for j in range(len(sent) + 1):
                            probabilities[i].append((j, parse_output_heads.data[i][j]))
                        probabilities[i].sort(key=lambda pair: pair[1], reverse=True)

                        maximum, indices = parse_output_heads[i].max(0)
                        predicted = indices.data[0]
                        expected = parse_head_targets.data[i]
                        head = predicted
                        heads[i] = head
                        is_head_correct = expected == predicted
                        if is_head_correct:
                            parse_scores.num_unlabeled_arcs += 1
                        else:
                            is_sent_correct = False
                            is_labeled_sent_correct = False

                        maximum, indices = parse_output_rels[i].max(0)
                        predicted = indices.data[0]
                        expected = parse_rel_targets.data[i]
                        rel = self.index2rel[predicted]
                        rels[i] = rel
                        if expected == predicted:
                            parse_scores.num_labels += 1
                            if is_head_correct:
                                parse_scores.num_labeled_arcs += 1
                        else:
                            is_labeled_sent_correct = False

            # Specially for parser
            if parse_scores is not None:
                fout.write("#text = {}\n".format(" ".join(forms)))

                if is_sent_correct:
                    parse_scores.num_unlabeled_trees += 1
                if is_labeled_sent_correct:
                    parse_scores.num_labeled_trees += 1

                parse_output_heads = parse_output_heads.data
                # Trying to turn random graph into well-formed tree.
                # Step 1. Find a root

                outliers = set()
                roots = set()
                unvisited = set()
                maximum = parse_output_heads[0][0] - 1.0
                index = -1
                for i in range(len(sent)):
                    if parse_output_heads[i][0] > maximum:
                        maximum = parse_output_heads[i][0]
                        index = i
                    if heads[i] == 0 or rels[i] == self.root_tag:
                        roots.add(i)
                    else:
                        unvisited.add(i)
                roots.add(index)

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
                    raise RuntimeError("unreachable branch")
                    # maximum = parse_output_heads[0][0] - 1.0
                    # index = -1
                    # for i in range(len(sent)):
                    #     if parse_output_heads[i][0] > maximum:
                    #         maximum = parse_output_heads[i][0]
                    #         index = i
                    # root = index
                    # if root in outliers:
                    #     outliers.remove(root)
                    # if root in unvisited:
                    #     unvisited.remove(root)
                    # root, visited, outliers = try_build_tree(root, heads, unvisited, outliers)
                heads[root] = 0
                rels[root] = self.root_tag

                # Now 'unvisited' contains only unresolved references.
                # Use minimal algo to resolve arcs
                while len(outliers) > 0:
                    options = []
                    for node in outliers:
                        options.append(try_expand_tree(node, probabilities, heads, visited, outliers))
                    options.sort(key=lambda t: len(t[3]))
                    index, head, visited, outliers = options[0]
                    heads[index] = head

                is_modified_sent_correct = True
                for i in range(len(sent)):
                    if heads[i] == parse_head_targets.data[i]:
                        parse_scores.num_modified_unlabeled_arcs += 1
                    else:
                        is_modified_sent_correct = False
                    fout.write("{}\t{}\t{}\t{}\t_\t_\t{}\t{}\t{}:{}\t_\n".format(
                        i + 1, forms[i], words[i], tags[i], heads[i], rels[i], heads[i], rels[i]))
                fout.write("\n")

                if is_modified_sent_correct:
                    parse_scores.num_modified_unlabeled_trees += 1


            # Specially for chunker determine the quantity of retrieved relevant chunks
            # if chunk_scores is not None:
            #     gold_chunks = [chunk for word, tag, chunk in sent]
            #     num_retrieved_relevant = 0
            #     i = 0
            #     while i < len(gold_chunks):
            #         gold_chunk = gold_chunks[i]
            #         actual_chunk = actual_chunks[i]
            #
            #         if gold_chunk[0] == "B":
            #             is_correct = True
            #             while True:
            #                 if gold_chunk != actual_chunk:
            #                     is_correct = False
            #
            #                 i += 1
            #                 if i == len(gold_chunks):
            #                     break
            #
            #                 gold_chunk = gold_chunks[i]
            #                 actual_chunk = actual_chunks[i]
            #
            #                 if gold_chunk[0] != "I":
            #                     if actual_chunk[0] == "I":
            #                         is_correct = False
            #                     break
            #
            #             if is_correct:
            #                 num_retrieved_relevant += 1
            #         else:
            #             i += 1
            #     chunk_scores.num_retrieved_relevant += num_retrieved_relevant

            # Emulate progress bar
            current_word += len(sent)
            if current_word >= next_interval:
                next_interval += words_per_interval
                print('💪', end="", flush=True)

        # Debug epoch log
        print("], ATL: {:10.8f}, ACL: {:10.8f}, APL: {:10.8f}".format(
            tag_loss / len(sents), chunk_loss / len(sents), parse_loss / len(sents)))
        fout.close()

    ##########################################################################
    def print_vectorizer_misses(self):
        print("unknown words: {} from {} ({:4.2f}%)".format(
            self.vectorizer.num_word_misses, self.vectorizer.num_words,
            self.vectorizer.num_word_misses / self.vectorizer.num_words * 100.0
        ))
        print("unknown grams: {} from {} ({:4.2f}%)".format(
            self.vectorizer.num_char_misses, self.vectorizer.num_grams,
            self.vectorizer.num_char_misses / self.vectorizer.num_grams * 100.0
        ))

    ##########################################################################
    @staticmethod
    def get_num_words(sents):
        num_words = 0
        for sent in sents:
            num_words += len(sent)
        return num_words

    ##########################################################################
    def create_or_load_indices(self):
        tag_path = "{}/{}_tags.txt".format(self.path_base, self.name)
        chunk_path = "{}/{}_chunks.txt".format(self.path_base, self.name)
        rel_path = "{}/{}_rels.txt".format(self.path_base, self.name)

        create_tag_index = False
        create_chunk_index = False
        create_rel_index = False

        # Try load
        if os.path.exists(tag_path):
            # Load from existing data base
            self.log("Loading POS tag index from file")

            for line in open(tag_path):
                tag, index = line.split()
                index = int(index)
                self.tags.add(tag)
                self.tag2index[tag] = index
                self.index2tag[index] = tag

            self.num_tags = len(self.tags)
        else:
            # Create from scratch
            self.log("Creating POS tag index")
            create_tag_index = True

        # Try load
        if os.path.exists(chunk_path):
            # Load chunk index from file
            self.log("Loading chunk index from file")

            for line in open(chunk_path):
                chunk, index = line.split()
                index = int(index)
                self.chunks.add(chunk)
                self.chunk2index[chunk] = index
                self.index2chunk[index] = chunk

            self.num_chunks = len(self.chunks)
        else:
            # Create from scratch
            self.log("Creating chunk tag index")
            create_chunk_index = True

        # Try load
        if os.path.exists(rel_path):
            # Load rel index from file
            self.log("Loading rel index from file")

            for line in open(rel_path):
                rel, index = line.split()
                index = int(index)
                self.rels.add(rel)
                self.rel2index[rel] = index
                self.index2rel[index] = rel

            self.num_rels = len(self.rels)
        else:
            # Create from scratch
            self.log("Creating rel tag index")
            create_rel_index = True

        # Create if necessary
        if create_tag_index or create_chunk_index or create_rel_index:
            # Collect data
            for sent in self.sents:
                self.num_words += len(sent)
                for dep, rel, head in sent:
                    form, word, tag = dep
                    if create_tag_index:
                        self.tags.add(tag)
                    if create_rel_index:
                        self.rels.add(rel)
                    # if create_chunk_index:
                    #     self.chunks.add(chunk)

            # Create POS tag database
            if create_tag_index:
                file_tags = open(tag_path, "w")
                self.num_tags = len(self.tags)
                for index, tag in enumerate(self.tags):
                    self.index2tag[index] = tag
                    self.tag2index[tag] = index
                    file_tags.write("{} {}\n".format(tag, index))
                file_tags.close()

            # Create chunk tag database
            if create_chunk_index:
                file_chunks = open(chunk_path, "w")
                self.num_chunks = len(self.chunks)
                for index, chunk in enumerate(self.chunks):
                    self.index2chunk[index] = chunk
                    self.chunk2index[chunk] = index
                    file_chunks.write("{} {}\n".format(chunk, index))
                file_chunks.close()

            # Create rel tag database
            if create_rel_index:
                file_rels = open(rel_path, "w")
                self.num_rels = len(self.rels)
                for index, rel in enumerate(self.rels):
                    self.index2rel[index] = rel
                    self.rel2index[rel] = index
                    file_rels.write("{} {}\n".format(rel, index))
                file_rels.close()

    ##########################################################################
    def log(self, message):
        print("Model [{}]:".format(self.name), message)
