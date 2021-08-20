import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(2,4) ,sharex=True, sharey=True)

y1 = np.arange(0,50, 0.5)
y2 = np.append(np.arange(0,20, 0.4), np.zeros(50))

axes[0].plot(y1)

axes[1].plot(y2)

axes[0].axvline(50, linestyle=':', color='black', linewidth=1)
axes[1].axvline(50, linestyle=':', color='black', linewidth=1)

axes[0].set_title('DUPLICATE_CHECK')
axes[0].set_facecolor('#ffd9d9')


axes[1].set_title('AUTOCOMMIT')
axes[1].set_facecolor('#d9d9ff')

plt.xlabel('number of insertions')
fig.text(-0.13, 0.5, 'performance influence in milliseconds', va='center', rotation='vertical')


plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

plt.savefig('influences.pdf', bbox_inches='tight')