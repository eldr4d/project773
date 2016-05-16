import numpy as np
import matplotlib.pyplot as plt
import public_variables
import csv

#POS=["$","V","U","O","&","R","L","~"]

POS=["Z","X"]



rows = []
with open(public_variables.POS_STATS, 'r') as fin:
    r=csv.DictReader(fin)
    for row in r:
        if(row["POS"] in POS):
            rows.append(row)

poss = [public_variables.POS_TAGSET[row["POS"]] for row in rows]
pmeans = [float(row["Schizophrenia mean"]) for row in rows]
nmeans = [float(row["Control mean"]) for row in rows]
pstd = [float(row["Schizophrenia stdev"]) for row in rows]
nstd = [float(row["Control stdev"]) for row in rows]

N = len(rows)
ind = np.arange(N)
width=0.35

fig, ax = plt.subplots()

rects1 = ax.bar(ind, nmeans,                  # data
                width,                          # bar width
                color='MediumSlateBlue',        # bar colour
                yerr=nstd,                  # data for error bars
                error_kw={'ecolor':'Tomato',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, pmeans,
                width,
                color='Tomato',
                yerr=pstd,
                error_kw={'ecolor':'MediumSlateBlue',
                          'linewidth':2})


axes = plt.gca()
ax.set_ylabel('Frequency')
ax.set_xlabel('Part of Speech')
#ax.set_title('Parts of Speech by Condition')
ax.set_xticks(ind + width)
ax.set_xticklabels(poss)
ax.legend((rects1[0], rects2[0]), ('Control','Schizophrenia'))
ax.set_ylim(ymin=0)
plt.show()