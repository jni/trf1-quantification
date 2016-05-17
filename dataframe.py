import pandas as pd
from skimage import io
import trf1 as tr
import matplotlib as mpl
kd_ims = io.imread_collection('../0.2uM APH N5/*.tif')
con_ims = io.imread_collection('../control N5/*.tif')
mpl.style.use('ggplot')
colnames = ['filename', 'file-number', 'aukb-kd', 'size(pixels)',
            'raw-mean', 'raw-total', 'raw-max', 'pre-mean', 'pre-total',
            'pre-max', 'post-mean', 'post-total', 'post-max', 'eccentricity']
kds = map(tr.trf_quantify, kd_ims)
cos = map(tr.trf_quantify, con_ims)
result = []
for i, (kd_fn, kd) in enumerate(zip(kd_ims.files, kds)):
    for blob_data in kd:
        result.append([kd_fn, i, 'kd'] + list(blob_data))
for j, (con_fn, co) in enumerate(zip(con_ims.files, cos), start=len(kd_ims)):
    for blob_data in co:
        result.append([con_fn, j, 'con'] + list(blob_data))
df = pd.DataFrame(result, columns=colnames)
df.to_csv('full-dataset-aph vs control trf1 N5.csv')
