import csv
import random
import torch
import torch.nn.functional as F
import json
import multiprocessing as mp

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

def transform_data(dlist, context_length, cdict):
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
    g = torch.Generator().manual_seed(220284)

    #initialising weights: tanh(embedding @ W1 + B1) * W2 + B2, ie instantiating model 
    embedding = torch.randn((char_count, embedding_dimension), requires_grad=True, generator=g)
    W1 = torch.randn((context_length * embedding_dimension, hidden_layer_1), requires_grad=True, generator=g)
    B1 = torch.randn(hidden_layer_1, requires_grad=True, generator=g)
    W2 = torch.randn((hidden_layer_1, char_count), requires_grad=True, generator=g)
    B2 = torch.randn(char_count, requires_grad=True, generator=g)

    return embedding, W1, B1, W2, B2

#training
def training_loop(train_dataset, eval_dataset, model, epochs, minibatch_size, context_length, embedding_dimension, learning_rate):
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
        weight_sum = 0
        for p in parameters:
            weight_sum += 0.01*(p**2).mean()
        loss = F.cross_entropy(logits, Ytr[indices]) + weight_sum #class indices

        #backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        if i > epochs * 2 / 3:
            learning_rate = 0.01
        elif i > epochs / 3:
            learning_rate = 0.05
        
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

    

def generate_names(model, context_length, embedding_dimension, idict):
    g = torch.Generator().manual_seed(220284)
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

def train_model(context_length, embedding_dimension, hidden_layer_1, learning_rate, epochs, minibatch_size):
    torch.set_num_threads(3)

    #constants
    char_count = 27
    datapath = "../names.txt"
    alphabets = ".abcdefghijklmnopqrstuvwxyz"
    cdict = {alphabets[i]: i for i in range(len(alphabets))}
    idict = {i: alphabets[i] for i in range(len(alphabets))}

    #preparing dataset
    train_data, eval_data, test_data = read_data(datapath)
    train_dataset, eval_dataset, test_dataset = transform_data(train_data, context_length, cdict), transform_data(eval_data, context_length, cdict), transform_data(test_data, context_length, cdict)
            
    #training
    model = create_model(char_count, embedding_dimension, context_length, hidden_layer_1)
    steps, tlosses, elosses= training_loop(train_dataset, eval_dataset, model, epochs, minibatch_size, context_length, embedding_dimension, learning_rate)

    #generation
    nlist, elist = generate_names(model, context_length, embedding_dimension, idict)
    pdict = {
        "steps": steps,
        "training_loss": tlosses,
        "evaluation_loss": elosses,
        "generated_names": nlist,
        "entropy_values": elist 
        }

    with open(f"{context_length}_{embedding_dimension}_{hidden_layer_1}_{learning_rate}_{epochs}_{minibatch_size}.json", "w") as file:
        json.dump(pdict, file)
            


if __name__ == "__main__":
    random.seed(42) 
    NUM_PROCESSES = 3
    argument_tasks = []

    with open('hyperparameters.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            #reading in hyperparameters
            context_length = int(row['context_length'])
            embedding_dimension = int(row['embedding_dimension'])
            hidden_layer_1 = int(row['hidden_layer_1'])
            learning_rate = float(row['learning_rate'])
            epochs = int(row['epochs'])
            minibatch_size = int(row['minibatch_size'])

            task_args = (context_length, embedding_dimension, hidden_layer_1, learning_rate, epochs, minibatch_size)
            argument_tasks.append(task_args)
    
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        pool.starmap(train_model, argument_tasks)


   #8, 6, 100, 0.1, 200000, 32 generates the most name like names



    
    