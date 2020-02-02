import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

train = pd.read_csv("data/train.csv")


def data_study():
    names = ['A', 'L4', 'L5', 'L6', 'L7', 'NONE', 'PC', 'SC', 'STON']
    pattern_SC = "^(SC)"
    pattern_A = "^(A)"
    pattern_PC = "^(PC)"
    pattern_STON = "^(SOTON)|^(STON)"
    pattern_length_4 = "^[0-9]{3,4}$"
    pattern_length_5 = "^[0-9]{5}$"
    pattern_length_6 = "^[0-9]{6}$"
    pattern_length_7 = "^[0-9]{7,9}$"
    ticket = train['Ticket']
    train['ticket_new'] = np.select(
        [ticket.str.contains(pattern_STON, regex=True), ticket.str.contains(pattern_SC, regex=True),
         ticket.str.contains(pattern_A, regex=True),
         ticket.str.contains(pattern_PC, regex=True), ticket.str.contains(pattern_length_4, regex=True),
         ticket.str.contains(pattern_length_5, regex=True),
         ticket.str.contains(pattern_length_6, regex=True), ticket.str.contains(pattern_length_7, regex=True)],
        ['STON', 'SC', 'A', 'PC', 'L4', 'L5', 'L6', 'L7'], default='NONE')
    values = list(train.groupby('ticket_new')['Survived'].mean())
    print(train.groupby('ticket_new')['Survived'].mean())
    plt.bar(names, values)
    plt.savefig('graphics/tickets.png')
    plt.show()


data_study()
