import json 
import math
import matplotlib.pyplot as plt
import os

def smooth_loss_and_compute_noise(values):
    #compute ema & use that value to compute RMSE for noise
    samples = 25
    beta = 0.96

    #compute for training loss
    ema_values, rmse_values = [0] * (samples - 1), [0] * (samples - 1)
    ema_curr = sum(values[:samples]) / samples
    ema_values.append(ema_curr)
    rmse_values.append(math.sqrt((ema_values[-1] - values[samples]) ** 2))
    for i in range(samples, len(values)):
        ema_curr = beta * ema_curr + (1 - beta) * values[i]
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

    #reading training data
    training_files = []
    directory = '/Users/euzhengxi/dev/deep-learning/makemore/part1'

    for filename in os.listdir(directory):
        if "json" in filename:
            file_path = os.path.join(directory, filename)
            training_files.append(file_path)

    #extracting useful information from training data
    steps = []
    tlosses_ema, tlosses_rmse = [], []
    elosses_ema, elosses_rmse = [], []
    names_distributions = []

    for filename in training_files:
        pdict = {}
        with open(filename, "r") as file:
            pdict = json.load(file)
        
        training_file = filename.split("/")[-1]
        steps = pdict["steps"]
        tlosses = pdict["training_loss"]
        elosses = pdict["evaluation_loss"]
        
        curr_tlosses_ema, curr_tlosses_rmse =  smooth_loss_and_compute_noise(tlosses)
        tlosses_ema.append((curr_tlosses_ema, training_file))
        tlosses_rmse.append((curr_tlosses_rmse, training_file))

        curr_elosses_ema, curr_elosses_rmse =  smooth_loss_and_compute_noise(elosses)
        elosses_ema.append((curr_elosses_ema, training_file))
        elosses_rmse.append((curr_elosses_rmse, training_file))

        nlist = pdict["generated_names"] 
        curr_names_distribution = compute_count(nlist)
        names_distributions.append((curr_names_distribution, training_file))

    #plotting the data
    plt.figure()
    for tloss, filename in tlosses_ema:
        plt.plot(steps, tloss, label=filename)
    plt.title('EMA smoothed training loss')
    plt.legend()
    plt.show()

    plt.figure()
    for tloss, filename in tlosses_rmse:
        plt.plot(steps, tloss, label=filename)
    plt.title('RMSE values: training')
    plt.legend()
    plt.show()

    plt.figure()
    for eloss, filename in elosses_ema:
        plt.plot(steps, eloss, label=filename)
    plt.title('EMA smoothed evaluation loss')
    plt.legend()
    plt.show()

    plt.figure()
    for eloss, filename in elosses_rmse:
        plt.plot(steps, eloss, label=filename)
    plt.title('RMSE values: evaluation')
    plt.legend()
    plt.show()

    plt.figure()
    for distribution, filename in names_distributions:
        plt.plot([i for i in range(11)], distribution, label=filename)
    plt.title('name distribution')
    plt.legend()
    plt.show()

    

