"""
Lab 3: Seq2Seq Translation — Simple RNN vs Attention Mechanism
==============================================================
实现基于RNN的Seq2Seq翻译模型，并添加Bahdanau注意力机制进行对比。

数据集: eng-fra.txt (英语-法语翻译对)
任务: 法语 → 英语翻译

运行方式:
    python seq2seq_translation.py
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import time
import math
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
hidden_size = 128
batch_size = 32
n_epochs = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Data Preparation
# ============================================================================

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    data_path = os.path.join(SCRIPT_DIR, 'data', '%s-%s.txt' % (lang1, lang2))
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs


# ============================================================================
# Models
# ============================================================================

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    """Simple RNN decoder WITHOUT attention."""
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long,
                                     device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention.

    energy = Va * tanh(Wa * keys + Ua * query)
    attention = softmax(energy)
    context = sum(attention * keys)
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [batch, 1, hidden]  keys: [batch, seq_len, hidden]
        scores = self.Va(torch.tanh(self.Wa(keys) + self.Ua(query)))
        # scores: [batch, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1)
        # attn_weights: [batch, seq_len, 1]
        context = torch.bmm(attn_weights.transpose(1, 2), keys)
        # context: [batch, 1, hidden]
        return context, attn_weights


class AttnDecoderRNN(nn.Module):
    """Attention-based decoder."""
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long,
                                     device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            # attn_weights: [batch, seq_len, 1] → squeeze → [batch, seq_len]
            attentions.append(attn_weights.squeeze(-1))

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # stack: list of [batch, seq_len] → [batch, MAX_LENGTH, seq_len]
        attentions = torch.stack(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights


# ============================================================================
# Training
# ============================================================================

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden,
                                         target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=50, plot_every=50):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100,
                                         print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0],
                                    input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# ============================================================================
# Visualization
# ============================================================================

def plot_loss_curve(losses, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', markersize=3)
    plt.xlabel('Epoch (per 50)')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_loss_comparison(losses_rnn, losses_attn, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_rnn, label='Seq2Seq w/o Attention', marker='o',
             markersize=3)
    plt.plot(losses_attn, label='Seq2Seq w/ Attention (Bahdanau)',
             marker='s', markersize=3)
    plt.xlabel('Epoch (per 50)')
    plt.ylabel('Loss')
    plt.title('Training Loss: Simple RNN Decoder vs Attention Decoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def showAttention(input_sentence, output_words, attentions, filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    input_words = input_sentence.split(' ')
    ax.set_xticks(range(attentions.shape[1]))
    ax.set_yticks(range(len(output_words)))
    ax.set_xticklabels(input_words + ['<EOS>'], rotation=90)
    ax.set_yticklabels(output_words)

    plt.xlabel('Input Words')
    plt.ylabel('Output Words')
    plt.title('Attention Visualization')
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Lab 3: Seq2Seq Translation — RNN vs Attention")
    print("=" * 60)

    # -- Experiment 1: Simple Seq2Seq (without Attention) --
    print("\n" + "=" * 60)
    print("Experiment 1: Simple Seq2Seq Model (without Attention)")
    print("=" * 60)

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

    encoder_rnn = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder_rnn = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    print(f"\nEncoder params: {sum(p.numel() for p in encoder_rnn.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder_rnn.parameters()):,}")
    print(f"\n{encoder_rnn}")
    print(f"\n{decoder_rnn}")

    print("\nTraining Simple RNN Seq2Seq...")
    losses_rnn = train(train_dataloader, encoder_rnn, decoder_rnn, n_epochs,
                       print_every=50, plot_every=50)
    plot_loss_curve(losses_rnn, "Training Loss: Seq2Seq without Attention",
                    os.path.join(output_dir, "loss_rnn.png"))

    print("\n--- Evaluation: Simple RNN Seq2Seq ---")
    encoder_rnn.eval()
    decoder_rnn.eval()
    evaluateRandomly(encoder_rnn, decoder_rnn, pairs, input_lang, output_lang)

    # -- Experiment 2: Seq2Seq with Bahdanau Attention --
    print("\n" + "=" * 60)
    print("Experiment 2: Seq2Seq Model WITH Bahdanau Attention")
    print("=" * 60)

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

    encoder_attn = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder_attn = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    print(f"\nEncoder params: {sum(p.numel() for p in encoder_attn.parameters()):,}")
    print(f"Attn Decoder params: {sum(p.numel() for p in decoder_attn.parameters()):,}")
    print(f"\n{encoder_attn}")
    print(f"\n{decoder_attn}")

    print("\nTraining Attention Seq2Seq...")
    losses_attn = train(train_dataloader, encoder_attn, decoder_attn, n_epochs,
                        print_every=50, plot_every=50)
    plot_loss_curve(losses_attn, "Training Loss: Seq2Seq with Attention",
                    os.path.join(output_dir, "loss_attn.png"))

    print("\n--- Evaluation: Attention Seq2Seq ---")
    encoder_attn.eval()
    decoder_attn.eval()
    evaluateRandomly(encoder_attn, decoder_attn, pairs, input_lang, output_lang)

    # -- Comparison --
    print("\n" + "=" * 60)
    print("Comparison: RNN vs Attention")
    print("=" * 60)

    plot_loss_comparison(losses_rnn, losses_attn,
                         os.path.join(output_dir, "loss_comparison.png"))

    test_sentences = [
        'il n est pas aussi grand que son pere',
        'je suis trop fatigue pour conduire',
        'je suis desole si c est une question idiote',
        'elle est en train de peindre un tableau',
        'nous sommes tous ensemble a l instant',
        'vous etes trop maigre',
        'tu es matinal',
        'je suis reellement fiere de vous',
    ]

    print("\nSide-by-side translation comparison:\n")
    for sent in test_sentences:
        words_rnn, _ = evaluate(encoder_rnn, decoder_rnn, sent,
                                 input_lang, output_lang)
        words_attn, attns = evaluate(encoder_attn, decoder_attn, sent,
                                      input_lang, output_lang)
        print(f"Input:        {sent}")
        print(f"No Attention: {' '.join(words_rnn)}")
        print(f"Attention:    {' '.join(words_attn)}")
        print("-" * 60)

    # -- Attention Visualization --
    print("\n" + "=" * 60)
    print("Attention Visualization")
    print("=" * 60)

    vis_sentences = [
        'il n est pas aussi grand que son pere',
        'je suis trop fatigue pour conduire',
        'je suis reellement fiere de vous',
    ]
    for sent in vis_sentences:
        output_words, attentions = evaluate(encoder_attn, decoder_attn, sent,
                                             input_lang, output_lang)
        print(f"\nInput:  {sent}")
        print(f"Output: {' '.join(output_words)}")
        safe_name = sent.replace(' ', '_')[:40]
        showAttention(sent, output_words,
                      attentions[0, :len(output_words), :],
                      filename=os.path.join(output_dir, f"attn_{safe_name}.png"))

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
