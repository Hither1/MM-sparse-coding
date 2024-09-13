import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


a1 = np.loadtxt('./vis/txt_phase1.txt', dtype=int)
b1 = np.loadtxt('./vis/img_phase1.txt', dtype=int)
a2 = np.loadtxt('./vis/dropout_0.2_txt_phase1.txt', dtype=int)

b2 = np.loadtxt('./vis/dropout_0.2_img_phase1.txt', dtype=int)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].hist(a1, bins=30, color='Yellow', edgecolor='black', log=True, alpha=0.9)
axes[0].hist(a2, bins=30, color='Pink', edgecolor='black', log=True, alpha=0.6)
axes[0].set_title('Histogram (texts)')

axes[1].hist(b1, bins=30, color='Yellow', edgecolor='black',log=True, alpha=0.9)
axes[1].hist(b2, bins=30, color='Pink', edgecolor='black',log=True, alpha=0.6)
axes[1].set_title('Histogram (images)')

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['Yellow', 'Pink']]
labels= ["Phase 1", "Phase 1 (add. dropout to MAE)"]
axes[0].legend(handles, labels)
axes[1].legend(handles, labels)

for ax in axes:
    ax.set_xlabel('Token activated times')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('./vis/' + 'hist.png')