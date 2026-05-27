import csv
import random
import torch
import torch.nn.functional as F
import json

#reading, transforming and preparing data
def read_data(dataPath):
    dlist = []
    with open(dataPath, 'r') as file:
        for line in file:
            dlist.append(line[:-1])
    
    random.shuffle(dlist)
    size = len(dlist)
    limit1, limit2 = int(0.8 * size), int(0.9 * size)
    train_data = dlist[:limit1]
    eval_data = dlist[limit1: limit2]
    test_data = dlist[limit2:]

    return train_data, eval_data, test_data

def transform_data(dlist, context_length):
    inputt = []
    output = []
    for word in dlist:
        context = [0] * context_length
        for char in word + '.':
            inputt.append(context)
            output.append(cdict[char])
            context = context[1:] + [cdict[char]]
    X = torch.tensor(inputt)
    Y = torch.tensor(output)
    return (X, Y)

def create_model(char_count, embedding_dimension, context_length, hidden_layer_1):
    #initialising weights: tanh(embedding @ W1 + B1) * W2 + B2, ie instantiating model 
    embedding = torch.randn((char_count, embedding_dimension), requires_grad=True, generator=g)
    W1 = torch.randn((context_length * embedding_dimension, hidden_layer_1), requires_grad=True, generator=g)
    B1 = torch.randn(hidden_layer_1, requires_grad=True, generator=g)
    W2 = torch.randn((hidden_layer_1, char_count), requires_grad=True, generator=g)
    B2 = torch.randn(char_count, requires_grad=True, generator=g)

    return embedding, W1, B1, W2, B2

#training
def train(train_dataset, eval_dataset, model, minibatch_size, context_length, embedding_dimension, learning_rate):
    Xtr, Ytr = train_dataset
    Xeval, Yeval = eval_dataset

    embedding, W1, B1, W2, B2 = model
    parameters = [embedding, W1, B1, W2, B2]

    #training loop
    steps = []
    tlosses = []
    elosses = []
    for i in range(epochs):
        steps.append(i)

        #training
        indices = torch.randint(low=0, high=Xtr.shape[0], size=(minibatch_size, )) #minibatch, context_length
        train_input = embedding[Xtr[indices]] #minibatch, context_length * embedding_dimension 
        train_input = torch.reshape(train_input, (-1, context_length * embedding_dimension))
        logits = F.tanh(train_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, Ytr[indices]) #class indices

        #backwards pass
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -learning_rate * p.grad
        
        tlosses.append(loss.log10().item())

        #eval
        indices = torch.randint(low=0, high=Xeval.shape[0], size=(minibatch_size, ))
        eval_input = embedding[Xeval[indices]]
        eval_input = torch.reshape(eval_input, (-1, context_length * embedding_dimension))
        logits = F.tanh(eval_input @ W1 + B1) @ W2  + B2
        loss = F.cross_entropy(logits, Yeval[indices])
        elosses.append(loss.log10().item())
    
    return steps, tlosses, elosses

    

def generate_names(model, context_length, embedding_dimension):
    eps = 1e-9
    embedding, W1, B1, W2, B2 = model
    nlist = []
    elist = []
    for i in range(50):
        context = [0] * context_length
        while True:
            context_tensor = torch.tensor([context])
            inputt = embedding[context_tensor]
            inputt = torch.reshape(inputt, (-1, context_length * embedding_dimension))
            logits = F.tanh(inputt @ W1 + B1) @ W2  + B2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]

            
            entropy = -torch.sum(probs * torch.log2(probs + eps), dim=-1)
            elist.append(entropy.mean().item())
            if ix == 0:
                break
        
        name = ""
        for idx in context:
            name += idict[idx]

        nlist.append(name)
    
    return nlist, elist
    


if __name__ == "__main__":
    random.seed(42) #will this be shared? is a good idea to bundle everything together? 
    g = torch.Generator().manual_seed(220284)
    char_count = 27

    datapath = "../names.txt"
    cdict = {'.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 
         'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
        }

    idict = {0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 
            15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'
            }

    train_data, eval_data, test_data = read_data(datapath)

    with open('hyperparameters.csv', mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            #reading in hyperparameters
            context_length, embedding_dimension, hidden_layer_1 = int(row['context_length']), int(row['embedding_dimension']), int(row['hidden_layer_1'])
            learning_rate, epochs, minibatch_size = float(row['learning_rate']), int(row['epochs']), int(row['minibatch_size'])

            #preparing dataset
            train_dataset, eval_dataset, test_dataset = transform_data(train_data, context_length), transform_data(eval_data, context_length), transform_data(test_data, context_length)
            
            #training
            model = create_model(char_count, embedding_dimension, context_length, hidden_layer_1)
            steps, tlosses, elosses= train(train_dataset, eval_dataset, model, minibatch_size, context_length, embedding_dimension, learning_rate)

            #generation
            nlist, elist = generate_names(model, context_length, embedding_dimension)
            pdict = {
                "steps": steps,
                "training_loss": tlosses,
                "evaluation_loss": elosses,
                "generated_names": nlist,
                "entropy_values": elist 
            }

            with open(f"{context_length}_{embedding_dimension}_{hidden_layer_1}_{learning_rate}_{epochs}_{minibatch_size}.json", "w") as file:
                json.dump(pdict, file)


    #redesign training pipeline to read from csv
    #modify training pipeline to output and store the values
    #set up multiprocessing for concurrent training sessions: 3



    
    