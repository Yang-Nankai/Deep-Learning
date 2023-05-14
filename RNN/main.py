# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/14 13:31
"""

from data import *
from model import *
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(category_tensor, line_tensor):
    optimizer.zero_grad()
    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

# Just return an output given a line
def evaluate(line_tensor):
    output = rnn(line_tensor)
    return output


if __name__ == '__main__':
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()
    hidden = 32
    learning_rate = 0.035
    # rnn = RNN(n_letters, args.hidden, n_categories)
    # rnn = LSTM(n_letters, hidden, n_categories)
    rnn = myLSTM(n_letters, hidden, n_categories)
    print(rnn)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    current_loss = 0
    all_losses = []
    all_accurancy = []
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    start = time.time()
	
    correct_num = 0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        guess, guess_i = categoryFromOutput(output)
        if guess == category:
            correct_num += 1

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            # guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            all_accurancy.append(correct_num / plot_every)
            correct_num = 0
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.title('validation_loss')
    plt.savefig('./validation_loss')
    plt.figure()
    plt.plot(all_accurancy)
    plt.title('validation_accuracy')
    plt.savefig('./validation_accuracy')

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    fig.savefig('./sphinx_gallery_image')




