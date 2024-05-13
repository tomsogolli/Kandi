# script_det.py
#####################
# 30.1.2024
# Tommi Tynkkynen
#####################
# Bayesian neural network for MNIST recognition

import time
import os
import nni
import torch
import torchbnn as bnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
import random as rd
import numpy as np

device = ("cpu"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
)
print(f"Using {device} device")


class BayesianNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=28*28, out_features=512),
                nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=512, out_features=512),
                nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




def train_one_epoch(epoch_index):
    running_cost = 0.
    last_cost = 0.
    start_t = time.time()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        # predict outputs
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss_ce = loss_fn(outputs, labels)
        loss_kl = kl_loss(model)
        cost = loss_ce + kl_rate*loss_kl
        cost.backward()
        # Adjust learning weights
        optimizer.step()
        # Save data
        running_cost += cost.item()
    avg_cost = running_cost / len(train_loader)
    print("Training this epoch took", round(time.time()-start_t), "seconds")
    print("Average cost per batch:", avg_cost)
    return avg_cost


model = BayesianNeuralNetwork().to(device)

learning_rate = 0.01
kl_rate = 0.1

loss_fn = torch.nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)

# MNIST data training and test
bs_train = 200
bs_test = 1000
tf = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))))
train_loader = DataLoader(datasets.MNIST('/files/', train=True, download=True, transform=tf),
        batch_size=bs_train, shuffle=True)

test_loader = DataLoader(datasets.MNIST('/files/', train=False, download=True, transform=tf),
    batch_size=bs_test, shuffle=True)


# train batch
train = enumerate(train_loader)
batch_idx, (train_data, train_targets) = next(train)

# test batch
test = enumerate(test_loader)
batch_idx, (test_data, test_targets) = next(test)

# random batch
random = enumerate(test_loader)
batch_idx, (random_data, random_targets) = next(test)

win_count = 0

for i in range(bs_test):
    x = test_data[i]
    target = test_targets[i].item()
    logits = model(x)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    result = y_pred.item()

    if target == result:
        win_count += 1



print("Without training")
print("Wins ", str(win_count)+"/"+str(bs_test))
no_train = win_count/bs_test
print("Win precent {}%".format(round(100*no_train, 2)))

costs = []

epoch_number = 0

epochs = 10

best_tcost = 1_000_000.

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

start_time = time.time()

for epoch in range(epochs):
    print('Epoch {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_cost = train_one_epoch(epoch_number)
    costs.append(avg_cost) 


    running_tcost = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    for i, tdata in enumerate(test_loader):
        tinputs, tlabels = tdata
        toutputs = model(tinputs)
        tloss_ce = loss_fn(toutputs, tlabels)
        tloss_kl = kl_loss(model)
        tcost = tloss_ce + kl_rate*tloss_kl
        running_tcost += tcost.item()



    avg_tcost = running_tcost / (i + 1)
    print('COST train {} valid {}'.format(avg_cost, avg_tcost))

    # Track best performance, and save the model's state
    if avg_tcost < best_tcost:
        best_tloss = avg_tcost
        model_path = 'BNNmodel_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


print("Whole training took", round(time.time()-start_time), "seconds")

win_count = 0



for i in range(bs_test):
    x = test_data[i]
    target = test_targets[i].item()
    logits = model(x)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    result = y_pred.item()

    if target == result:
        win_count += 1



print("")
print("With", epochs, "epochs")
print("Wins ", str(win_count)+"/"+str(bs_test))
print("Win precent {}%".format(round((win_count/bs_test)*100, 2)))

print("Increase after training", round((win_count/bs_test - no_train)*100, 2), "%")

x_plot = []
for i in range(epochs):
    x_plot.append(i+1)

# xnew = np.linspace(1, 10, 300)  

# spl = make_interp_spline(x_plot, losses, k=3)
# losses_smooth = spl(xnew)

plt.figure()
plt.bar(x_plot, costs)
plt.xlabel("Epochs")
plt.ylabel("Avg cost")
plt.show()

plt.figure()
plt.bar(x_plot[1:], costs[1:])
plt.xlabel("Epochs")
plt.ylabel("Avg cost")
plt.show()

# testing NN
print("")
print("Test random sample")
c = 1
while c!=0:
    print("")
    print("1) for random sample")
    print("2) Show 10 times when NN is wrong")
    print("0) end")
    c = input("Give choice: ")
    if c=="":
        print("Try again")
        continue
    c = int(c)
    if c==1:
        r = rd.randint(0, 999)
        x = random_data[r]
        target = random_targets[r].item()
        logits = model(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        max_prob, max_index = torch.max(pred_probab[0], dim=0)
        result = y_pred.item()
        fig = plt.figure()
        plt.imshow(random_data[r][0], cmap='gray', interpolation='none')
        plt.title("Target: {} NN:{}".format(random_targets[r], result))
        plt.show()
        fig
        print("NN is {}% sure its {}".format(max_prob.item()*100, result))
        if target == result:
            print("NN is right")
        else:
            print("NN is wrong")
    elif c==2:
        count = 0
        fig = plt.figure()
        for i in range(bs_test):
            x = test_data[i]
            target = test_targets[i].item()
            logits = model(x)
            pred_probab = nn.Softmax(dim=1)(logits)
            y_pred = pred_probab.argmax(1)
            result = y_pred.item()
            if target != result:
                count += 1
                fig.add_subplot(2,5,count)
                plt.imshow(test_data[i][0], cmap='gray', interpolation='none')
                title_text = "T:{}".format(test_targets[i]) + " NN:" + str(result)
                plt.title(title_text)
                plt.xticks([])
                plt.yticks([])
            if count>=10:
                plt.show()
                break
        fig
    elif c==0:
        break
    else:
        print("Try again")
print("Ending")

################## EOF ##############################1
    
