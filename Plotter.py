import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_feature(net, plotloader, device,dirname, epoch=0,plot_class_num=2, maximum=0, plot_quality=200):
    plot_features = []
    plot_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(plotloader):
            if batch_idx%10==0:
                #print(batch_idx)
                inputs = inputs.reshape(1,1,1,18)
                inputs, targets = inputs.to(device), targets.to(device)
                embed_fea, _ = net(inputs)
                try:
                    embed_fea = embed_fea.data.cpu().numpy()
                    targets = targets.data.cpu().numpy()
                except:
                    embed_fea = embed_fea.data.numpy()
                    targets = targets.data.numpy()

                plot_features.append(embed_fea)
                plot_labels.append(targets)

    plot_features = np.concatenate(plot_features, 0)
    plot_labels = np.concatenate(plot_labels, 0)

    # print(centroids)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    make_ = ['+', "^", "s","."]


    for label_idx in range(plot_class_num):
        features = plot_features[plot_labels == label_idx,:]
        maximum = min(maximum, len(features)) if maximum>0 else len(features)
        plt.scatter(
            features[0:maximum, 0],
            features[0:maximum, 1],
            c=colors[label_idx],
            s=20,
            marker=make_[label_idx],
        )

    # currently only support 10 classes, for a good visualization.
    # change plot_class_num would lead to problems.
    legends= ['0', '1', 'unknown', '3', '4', '5', '6', '7', '8', '9']
    plt.legend(legends[0:plot_class_num], loc='upper right')

    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight',dpi=plot_quality)
    plt.close()


