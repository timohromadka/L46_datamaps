import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree

def get_data_subset(
        indexes,
        variability,
        confidence,
        correctness,
        forgetfulness,

        p_easytolearn=0.0,
        p_ambiguous=0.0,
        p_hardtolearn=0.0,

        p_random=0.0,

        p_variability=0.0,
        selector_variability='top',
        p_confidence=0.0,
        selector_confidence='top',
        p_correctness=0.0,
        selector_correctness='top',
        p_forgetfulness=0.0,
        selector_forgetfulness='top'):
    """
    Function to select subsets of data based on various criteria.
    """
    print('Fetching indices of specified training dynamics subset.')

    length = len(indexes)
    coordinates = np.hstack([np.expand_dims(variability, -1), np.expand_dims(confidence, -1)])
    selected_indices = set()

    # Handling KNN based selectors
    if p_easytolearn or p_ambiguous or p_hardtolearn:
        kdt = KDTree(coordinates, metric='euclidean')
        if p_ambiguous:
            n_ambiguous = int(p_ambiguous * length)
            _, ambiguous_idx = kdt.query([[1, 0.5]], k=n_ambiguous)
            selected_indices.update(indexes[i] for i in ambiguous_idx.flatten())
        if p_easytolearn:
            n_easytolearn = int(p_easytolearn * length)
            _, easy2learn_idx = kdt.query([[0, 1]], k=n_easytolearn)
            selected_indices.update(indexes[i] for i in easy2learn_idx.flatten())
        if p_hardtolearn:
            n_hardtolearn = int(p_hardtolearn * length)
            _, hardtolearn_idx = kdt.query([[0, 0]], k=n_hardtolearn)
            selected_indices.update(indexes[i] for i in hardtolearn_idx.flatten())
        return list(selected_indices)

    # Handling random selection
    if p_random > 0:
        n_random = int(p_random * length)
        random_indices = np.random.choice(indexes, n_random, replace=False)
        selected_indices.update(random_indices)
        return list(selected_indices)

    # Helper function for variability, confidence, correctness, forgetfulness
    def select_top_or_bottom(data, p_value, selector, indexes):
        if p_value > 0:
            n_select = int(p_value * length)
            sorted_idx = np.argsort(data)
            if selector == 'top':
                selected = sorted_idx[-n_select:]
            else:  # bottom
                selected = sorted_idx[:n_select]
            return set(indexes[i] for i in selected)
        return set()

    # Selecting based on variability, confidence, correctness, forgetfulness
    selected_indices.update(select_top_or_bottom(variability, p_variability, selector_variability, indexes))
    selected_indices.update(select_top_or_bottom(confidence, p_confidence, selector_confidence, indexes))
    selected_indices.update(select_top_or_bottom(correctness, p_correctness, selector_correctness, indexes))
    selected_indices.update(select_top_or_bottom(forgetfulness, p_forgetfulness, selector_forgetfulness, indexes))

    return list(selected_indices)

def plot_datamap(
        var1, # x axis
        var2, # y axis
        var3, # (z axis)
        var1_name='variability',
        var2_name='confidence',
        var3_name='correctness',
        dataset_name='cifar10',
        model_name='cnn'
    ):
    graph_title = f"Data map for {dataset_name.upper()}, based on {model_name}. Variables: {var1_name}, {var2_name}, {var3_name}"

    # Creating a new figure with a 3x4 grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 4)

    # first 3x3 grid
    ax_main = fig.add_subplot(gs[:, :3])
    sns.scatterplot(x=var1, y=var2, hue=var3, palette="coolwarm", s=15, legend='auto', ax=ax_main, edgecolor='none', alpha=1)
    sns.kdeplot(x=var1, y=var2,
                levels=8, color=sns.color_palette("Paired")[7], linewidths=0.7, ax=ax_main)

    ax_main.set(title=graph_title,
                xlabel=var1_name, ylabel=var2_name,
                #xlim=(-0.005, 0.505), ylim=(-0.005, 1.005)
                )

    # Annotations
    box_style = {'boxstyle': 'round', 'facecolor': 'white', 'ec': 'black'}
    ax_main.text(0.14, 0.84, 'easy-to-learn', transform=ax_main.transAxes, verticalalignment='top', bbox=box_style)
    ax_main.text(0.75, 0.5, 'ambiguous', transform=ax_main.transAxes, verticalalignment='top', bbox=box_style)
    ax_main.text(0.14, 0.14, 'hard-to-learn', transform=ax_main.transAxes, verticalalignment='top', bbox=box_style)
    ax_main.legend(title=var3_name)

    # top-right grid
    ax_confidence = fig.add_subplot(gs[0, 3])
    sns.histplot(var1, bins=10, ax=ax_confidence, color='indigo')
    ax_confidence.set(title=f'{var1_name} distribution')

    # middle-right grid
    ax_variability = fig.add_subplot(gs[1, 3])
    sns.histplot(var2, bins=10, ax=ax_variability, color='teal')
    ax_variability.set(title=f'{var2_name} distribution')

    # bottom-right grid
    ax_correctness = fig.add_subplot(gs[2, 3])
    sns.histplot(var3, bins=10, ax=ax_correctness, color='mediumseagreen')
    ax_correctness.set(title=f'{var3_name} distribution')

    # Adjusting layout and saving the figure
    plt.tight_layout()

    # Saving fig
    output_dir = os.path.join('images', dataset_name) # if subdir is '', then no subdir created
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    graph_save_name = f"datamap_{model_name}_v1{var1_name}_v2{var2_name}_v3{var3_name}"
    image_save_file = os.path.join(output_dir, graph_save_name)
    #plt.savefig(image_save_file)
    
    return fig
