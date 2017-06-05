import os
import numpy as np
import pandas as pd
import imageio as iio
from skimage import filters
from glob import glob
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# Make image measurements
image_filenames = glob('*.tif')
classes = [fn.split()[2] for fn in image_filenames]
images = [iio.imread(fn) for fn in image_filenames]
reds = [image[..., 0] for image in images]
greens = [image[..., 1] for image in images]
redst = [r > filters.threshold_otsu(r) for r in reds]
greenst = [g > filters.threshold_otsu(g) for g in greens]
mean_red = [np.mean(red[rt]) for red, rt in zip(reds, redst)]
mean_green = [np.mean(green[gt]) for green, gt in zip(greens, greenst)]

# Put them in a dataframe and save it
df = pd.DataFrame({
    'filename': image_filenames,
    'class': classes,
    'red': mean_red,
    'green': mean_green
})
df.to_excel(os.path.expanduser('~') + '/Desktop/intensities.xlsx')

# Make a bar plot and save it
means = df.groupby('class').aggregate(np.mean)
errs = df.groupby('class').aggregate(np.std)
means.plot(kind='bar', yerr=errs)
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.expanduser('~') + '/Desktop/intensities.png',
            dpi=300)

# Make a jitter plot and save it
td = pd.melt(df, id_vars=['filename', 'class'],
             value_vars=['red', 'green'], var_name='channel',
             value_name='intensity').set_index('filename')
ax = sns.stripplot(x='class', y='intensity', hue='channel', data=td,
                   hue_order=('green', 'red'),
                   split=True, jitter=True)
ax.figure.savefig(os.path.expanduser('~') + '/Desktop/jitter.png',
                  dpi=300)
