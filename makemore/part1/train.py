import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATAPATH = "../names.txt"
CHAR_COUNT = 27
CONTEXT_LENGTH = 6
EMBEDDING_DIMENSION = 6
HIDDEN_LAYER_1 = 100
LEARNING_RATE = 0.01
EPOCHS = 10000
MINIBATCH_SIZE = 64

cdict = {
    '.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 
    'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
}

#reading, transforming and preparing data
def read_data(dataPath):
    dlist = []
    with open(dataPath, 'r') as file:
        for line in file:
            dlist.append(line[:-1])
    return dlist

def transform_data(dlist):
    inputt = []
    output = []
    for word in dlist:
        context = [cdict['.']] * CONTEXT_LENGTH
        for char in word + '.':
            inputt.append(context)
            output.append(cdict[char])
            context = context[1:] + [cdict[char]]
    X = torch.tensor(inputt)
    Y = torch.tensor(output)
    return (X, Y)

def prepare_data(dlist):
    random.seed(42)
    random.shuffle(dlist)
    size = len(dlist)
    limit1, limit2 = int(0.8 * size), int(0.9 * size)
    train_dataset = transform_data(dlist[:limit1])
    eval_dataset = transform_data(dlist[limit1: limit2])
    test_dataset = transform_data(dlist[limit2:])

    return train_dataset, eval_dataset, test_dataset

#training
def train(train_dataset, eval_dataset):
    Xtr, Ytr = train_dataset
    Xeval, Yeval = eval_dataset

    #initialising weights: tanh(embedding @ W1 + B1) * W2 + B2, ie instantiating model 
    g = torch.Generator().manual_seed(220284)
    embedding = torch.randn((CHAR_COUNT, EMBEDDING_DIMENSION), requires_grad=True, generator=g)
    W1 = torch.randn((CONTEXT_LENGTH * EMBEDDING_DIMENSION, HIDDEN_LAYER_1), requires_grad=True, generator=g)
    B1 = torch.randn(HIDDEN_LAYER_1, requires_grad=True, generator=g)
    W2 = torch.randn((HIDDEN_LAYER_1, CHAR_COUNT), requires_grad=True, generator=g)
    B2 = torch.randn(CHAR_COUNT, requires_grad=True, generator=g)
    parameters = [embedding, W1, B1, W2, B2]

    #training loop
    steps = []
    tlosses = []
    elosses = []
    for i in range(EPOCHS):
        steps.append(i)

        #training
        indices = torch.randint(low=0, high=len(train_dataset), size=(MINIBATCH_SIZE, )) #minibatch, 6
        train_input = embedding[Xtr[indices]] #minibatch, 6, 6
        train_expected = Ytr[indices] #class indices work
        train_input = torch.reshape(train_input, (-1, CONTEXT_LENGTH * EMBEDDING_DIMENSION))
        logits = F.tanh(train_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, train_expected) 
        tlosses.append(loss.item())

        #backwards pass
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -LEARNING_RATE * p.grad

        #eval
        indices = torch.randint(low=0, high=len(eval_dataset), size=(MINIBATCH_SIZE, ))
        eval_input = embedding[Xeval[indices]]
        eval_expected = Yeval[indices] 
        eval_input = torch.reshape(eval_input, (-1, CONTEXT_LENGTH * EMBEDDING_DIMENSION))
        logits = F.tanh(eval_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, eval_expected)
        elosses.append(loss.item())

    plt.plot(steps, tlosses, label="training loss", color="blue")
    plt.plot(steps, elosses, label="eval loss", color="orange")
    plt.show()


if __name__ == "__main__":
    names = read_data(DATAPATH)
    train_dataset, eval_dataset, test_dataset = prepare_data(names)
    print(f"train: {len(train_dataset[0])}, eval: {len(eval_dataset[0])}, test: {len(test_dataset[0])}")
    train(train_dataset, eval_dataset)


    
    