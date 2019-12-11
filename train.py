import torch
import torch.nn as nn
import numpy as np

from collections import Counter
import os
import io
from argparse import Namespace
from RNN import RNN

values = Namespace(
    train_file='harry1.txt',
    seq_size=16,
    batch_size=16,
    embedding_size=128,
    lstm_size=128,
    gradients_norm=5,
    initial_words=['Harry', 'Potter'],
    predict_top_k=5,
    sentence_num=5,
)


def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.split()

    print('Word count: ', len(text))
    # count every word and save count
    word_counts = Counter(text)
    # sort them with the quantity
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # todo: remove sightns like '-'
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    vocab_size = len(int_to_vocab)

    print('Vocabulary size', vocab_size)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    # cut the rest
    in_text = int_text[:num_batches * batch_size * seq_size]
    # move out by one
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, vocab_size, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    # print('in text: ',len(in_text),len(in_text[0]))
    for i in range(0, len(in_text[0]), seq_size):
        yield in_text[:, i:i + seq_size], out_text[:, i:i + seq_size]


def get_loss_and_train_op(net, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def train(net, criterion, optimizer, n_vocab, in_text, out_text, vocab_to_int, int_to_vocab, device, epochs=200):
    iteration = 0

    for epochs in range(epochs):
        print("Epoch: %d" % (epochs + 1))
        state_h = net.zero_state(values.batch_size).to(device)
        state_c = net.zero_state(values.batch_size).to(device)
        for x, y in get_batches(in_text, out_text, values.batch_size, values.seq_size):
            iteration += 1
            net.train()

            # print('batch: ',x.shape)
            optimizer.zero_grad()

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            output, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(output.transpose(1, 2), y)

            loss_value = loss.item()

            loss.backward()

            # IMPORTANT for autograd. idky
            state_h = state_h.detach()
            state_c = state_c.detach()

            # gradient clipping
            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), values.gradients_norm)

            optimizer.step()

        print('Loss: {}'.format(loss_value))
        if epochs % 10 == 0:
            predict(device, net, values.initial_words, n_vocab, vocab_to_int,
                    int_to_vocab, values.predict_top_k)

    predict(device, net, values.initial_words, n_vocab, vocab_to_int,
            int_to_vocab, values.predict_top_k)

    return net


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=values.predict_top_k):
    net.eval()
    # words = flags.initial_words
    words = ['Harry', 'Potter']
    # words = ['Sehr']

    state_h = net.zero_state(1).to(device)
    state_c = net.zero_state(1).to(device)

    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    number_of_sent = 0
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        # -----test
        if choice.contains('.'):
            number_of_sent = number_of_sent + 1
            if number_of_sent >= 5:
                break
        # ------test
        words.append(int_to_vocab[choice])

    print(' '.join(words))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        values.train_file, values.batch_size, values.seq_size)

    net = RNN(n_vocab, values.embedding_size, values.lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.001)

    net = train(net, criterion, optimizer, n_vocab, in_text, out_text, vocab_to_int, int_to_vocab, device)

    torch.save(net,'/data/myNet.pt')

if __name__ == '__main__':
    main()
