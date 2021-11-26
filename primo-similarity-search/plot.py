# https://github.com/YisongMiao/text-dna/blob/main/primo-similarity-search/notebooks/03_simulation/03_plot_results.ipynb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# yields = pd.read_hdf('../data/simulation/extended_targets/callie_janelle.h5')
# dists = pd.read_hdf('../data/extended_targets/query_target_dists.h5')

yields = pd.read_hdf('../data/simulation/extended_targets/steve-self.h5')
# yields = pd.read_hdf('../data/simulation/extended_targets/callie_janelle.h5')


dists = pd.read_hdf('../data/targets/query_target_dists-text.h5')

# df = yields.join(dists['callie_janelle'].rename('euclidean_distance'))
df = yields.join(dists['plain'].rename('euclidean_distance'))


# Color here represents density.
# For more info on Hexbins, check out https://holypython.com/python-visualization-tutorial/creating-hexbin-charts/
plt.hexbin(df.euclidean_distance, df.duplex_yield, gridsize=50, bins='log')
plt.show()

thresholds = [1.1, 1.2, 1.3, 1.4]
bin_labels = np.array(
    ["$\leq%d$" % thresholds[0]]
    + ["(%d, %d]" % (a,b) for (a,b) in zip(thresholds,thresholds[1:])]
    + [">%d" % thresholds[-1]]
)

plt.figure(figsize = (5, 3), dpi=150)
(lambda data:
    sns.violinplot(

        x='dist_bin',
        y='duplex_yield',
        data=data,
        linewidth=0.4,
        fliersize=0.5,
        cut=0.0,
        scale='width',
        order = bin_labels
    )
)(
    df
    .assign(
        dist_bin = lambda df: bin_labels[np.digitize(df.euclidean_distance, thresholds, right=True)]
    )
)
plt.xlabel("Euclidean distance")
plt.xticks(rotation=20)
[label.set_fontsize(12) for label in plt.gca().get_xticklabels()]
plt.ylabel("Simulated yield")
plt.tight_layout()
plt.show()


if __name__ == '__main__':
    print 'Done'
