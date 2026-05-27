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
    #smoothed out train and eval loss, noise, output lnegth probability distribution
    #entropy? 

    pdict = {}
    with open("6_15_200_0.1_100_64.json", "r") as file:
        pdict = json.load(file)
    
    steps = pdict["steps"]
    tlosses = pdict["training_loss"]
    elosses = pdict["evaluation_loss"]
    tlosses_ema, tlosses_rmse =  smooth_loss_and_compute_noise(tlosses)
    elosses_ema, elosses_rmse =  smooth_loss_and_compute_noise(elosses)
    plt.plot(steps, tlosses_ema, color="blue")
    plt.plot(steps, tlosses_rmse, color="orange")
    plt.plot(steps, elosses_ema, color="green")
    plt.plot(steps, elosses_rmse, color="black")
    plt.show()

    plt.figure()
    nlist = pdict["generated_names"] #bar chart for count for each variation
    names_distribution = compute_count(nlist)
    plt.bar([i for i in range(11)], names_distribution)
    plt.show()

    #plt.figure()
    #elist = pdict["entropy_values"] #is this useful? how should i make sense of it? 

    

