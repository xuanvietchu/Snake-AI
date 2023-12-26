import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, n_games, file_name):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(n_games-1, scores[-1], str(scores[-1]))
    plt.text(n_games-1, mean_scores[-1], str(mean_scores[-1]))

    if file_name:
        plt.savefig(file_name)
    else:
        plt.pause(0.1)
