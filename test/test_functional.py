import matplotlib.pyplot as plt 

def test_data_set(data_set, plot_samples = 100):
    paths, information, payoff = data_set 
    print('Shape of paths: ', paths.shape)
    print('Shape of information: ', information.shape)
    print('Shape of payoff: ', payoff.shape)
    
    plt.figure
    plt.plot(paths[:plot_samples,:,0].T)
    plt.grid()
    plt.show()

    plt.figure
    plt.scatter(paths[:,-1,0], payoff[:,-1,0])
    plt.grid()
    plt.show()
