import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATAPATH = "../names.txt"
CHAR_COUNT = 27
CONTEXT_LENGTH = 6
EMBEDDING_DIMENSION = 15
HIDDEN_LAYER_1 = 200
LEARNING_RATE = 0.1
EPOCHS = 100000
MINIBATCH_SIZE = 64

cdict = {'.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 
         'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
        }

idict = {0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 
         15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'
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
        context = [0] * CONTEXT_LENGTH
        for char in word + '.':
            inputt.append(context)
            output.append(cdict[char])
            context = context[1:] + [cdict[char]]
    X = torch.tensor(inputt)
    Y = torch.tensor(output)
    return (X, Y)

def prepare_data(dlist):
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
        indices = torch.randint(low=0, high=Xtr.shape[0], size=(MINIBATCH_SIZE, )) #minibatch, 6
        train_input = embedding[Xtr[indices]] #minibatch, 6, 6
        train_input = torch.reshape(train_input, (-1, CONTEXT_LENGTH * EMBEDDING_DIMENSION))
        logits = F.tanh(train_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, Ytr[indices]) #class indices

        #backwards pass
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -LEARNING_RATE * p.grad
        
        tlosses.append(loss.log10().item())

        #eval
        indices = torch.randint(low=0, high=Xeval.shape[0], size=(MINIBATCH_SIZE, ))
        eval_input = embedding[Xeval[indices]]
        eval_input = torch.reshape(eval_input, (-1, CONTEXT_LENGTH * EMBEDDING_DIMENSION))
        logits = F.tanh(eval_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, Yeval[indices])
        elosses.append(loss.log10().item())

    plt.plot(steps, tlosses, label="training loss", color="blue")
    plt.plot(steps, elosses, label="eval loss", color="orange")
    plt.show()

    return embedding, W1, B1, W2, B2

def generate_names(embedding, W1, B1, W2, B2):
    for i in range(10):
        context = [0] * CONTEXT_LENGTH
        while True:
            context_tensor = torch.tensor([context])
            inputt = embedding[context_tensor]
            inputt = torch.reshape(inputt, (-1, CONTEXT_LENGTH * EMBEDDING_DIMENSION))
            logits = F.tanh(inputt @ W1 + B1) @ W2  + B2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
        
        name = ""
        for idx in context:
            name += idict[idx]
        print(name)

if __name__ == "__main__":
    g = torch.Generator().manual_seed(220284)
    random.seed(42)

    names = read_data(DATAPATH)
    train_dataset, eval_dataset, test_dataset = prepare_data(names)
    print(f"train: {len(train_dataset[0])}, eval: {len(eval_dataset[0])}, test: {len(test_dataset[0])}")
    embedding, W1, B1, W2, B2 = train(train_dataset, eval_dataset)
    generate_names(embedding, W1, B1, W2, B2)


    
    