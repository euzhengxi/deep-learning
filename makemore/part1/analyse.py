import json 
import math
import matplotlib.pyplot as plt

def smooth_loss_and_compute_noise(values):
    #compute ema & use that value to compute RMSE for noise
    beta = 0.9

    #compute for training loss
    ema_values, rmse_values = [0] * 9, [0] * 9
    ema_curr = sum(values[:10]) / 10
    ema_values.append(ema_curr)
    rmse_values.append(math.sqrt((ema_values[-1] - values[10]) ** 2))
    for i in range(10, len(values)):
        ema_curr = beta * ema_curr + (1 - 0.1) * values[i]
        rmse_curr = math.sqrt((ema_curr - values[i]) ** 2)
        ema_values.append(ema_curr)
        rmse_values.append(rmse_curr)
    
    return ema_values, rmse_values

def compute_count(names):
    count = [0] * 11
    for name in names:
        curr = len(name)
        for c in name:
            if c == '.':
                curr -= 1
        
        if curr < 10:
            count[curr] += 1
        else:
            count[10] += 1

    return count

if __name__ == "__main__":
    #performance indicators
    #smoothed out train and eval loss, noise, output length probability distribution
    #entropy? 

    #extracting useful information from training data
    training_files = ["6_15_200_0.1_100_64.json", "6_15_200_0.1_100_64.json", "6_15_200_0.1_100_64.json"]
    steps = []
    tlosses_ema, tlosses_rmse = [], []
    elosses_ema, elosses_rmse = [], []
    names_distributions = []

    for filename in training_files:
        pdict = {}
        with open(filename, "r") as file:
            pdict = json.load(file)
        
        steps = pdict["steps"]
        tlosses = pdict["training_loss"]
        elosses = pdict["evaluation_loss"]
        curr_tlosses_ema, curr_tlosses_rmse =  smooth_loss_and_compute_noise(tlosses)
        tlosses_ema.append(curr_tlosses_ema)
        tlosses_rmse.append(curr_tlosses_rmse)

        curr_elosses_ema, curr_elosses_rmse =  smooth_loss_and_compute_noise(elosses)
        elosses_ema.append(curr_elosses_ema)
        elosses_rmse.append(curr_elosses_rmse)

        nlist = pdict["generated_names"] 
        curr_names_distribution = compute_count(nlist)
        names_distributions.append(curr_names_distribution)

    #plotting the data
    plt.figure()
    for tloss in tlosses_ema:
        plt.plot(steps, tloss)
    plt.show()

    plt.figure()
    for tloss in tlosses_rmse:
        plt.plot(steps, tloss)
    plt.show()

    plt.figure()
    for eloss in elosses_ema:
        plt.plot(steps, eloss)
    plt.show()

    plt.figure()
    for eloss in elosses_rmse:
        plt.plot(steps, eloss)
    plt.show()

    plt.figure()
    for distribution in names_distributions:
        plt.plot([i for i in range(11)], distribution)
    plt.show()

    #plt.figure()
    #elist = pdict["entropy_values"] #is this useful? how should i make sense of it? 

    

