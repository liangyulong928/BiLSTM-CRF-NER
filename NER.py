import torch
import pickle

with open('./dataset/train.txt') as file:
     content = file.readlines()

message = []
tokens = []
tag = []

for line in content:
    if line == '\n':
        message.append((tokens,tag))
        tokens = []
        tag = []
    else:
        line = line.rstrip('\n')
        contents = line.split(' ')
        tokens.append(contents[0])
        tag.append(contents[-1])

words = []
labels = []
for i in range(len(message)):
    words += message[i][0]
    labels += message[i][1]

words_dict = list(set(words))
labels_dict = list(set(labels))
words_dict.append("<pad>")

message_index = []
labels_index = []
for msg in message:
    word_index = []
    label_index = []
    if len(msg[0]) < 37:
        pad_len = 37 - len(msg[0])
        for m in msg[0]:
            word_index.append(words_dict.index(m))
        word_index = word_index + [len(words_dict) - 1] * pad_len
        for l in msg[1]:
            label_index.append(labels_dict.index(l))
        label_index = label_index + [labels_dict.index('O')] * pad_len
    elif len(msg[0]) > 37:
        for m in msg[0]:
            word_index.append(words_dict.index(m))
        word_index = word_index[:36]
        for l in msg[1]:
            label_index.append(labels_dict.index(l))
        label_index = label_index[:36]
    message_index.append(torch.Tensor(word_index))
    labels_index.append(torch.Tensor(label_index))

torch.save(message_index,'./dataloader/message_tensor.pt')
torch.save(labels_index,'./dataloader/label_tensor.pt')

def get_label_dict():
    return dict({(labels_dict[i],i) for i in range(len(labels_dict))})
labels_tensor = get_label_dict()

with open("./dataloader/labels_dict.pkl", "wb") as tf:
    pickle.dump(labels_tensor,tf)

print(labels_tensor)