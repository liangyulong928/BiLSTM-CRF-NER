import torch
import model
import pickle
def load_mask(msg):
    masks = []
    for i in msg:
        if len(torch.nonzero(i==23623))!=0:
            length = int(torch.nonzero(i==23623)[0])
        else:
            length = 37
        mask = [1] * length + [0] * (37-length)
        masks.append(mask)
    return masks

msg = torch.load('./dataloader/message_tensor.pt')
label = torch.load('./dataloader/label_tensor.pt')
mask = torch.Tensor(load_mask(msg))
with open("./dataloader/labels_dict.pkl", "rb") as tf:
    label_dict = pickle.load(tf)

model_imp = model.BiLSTM_CRF(23624, label_dict, 9, 9,23623,8)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for i in range(len(msg)):
    print("begin")
    loss = model_imp(msg[i],label[i],mask[i])
    loss.backward()
    optimizer.step()