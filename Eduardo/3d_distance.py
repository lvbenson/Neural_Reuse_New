from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "orange", "red"])

thresholds = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
#thresholds.reverse()

reps = 100
nn5 = 2*5
impact5 = np.zeros((reps,nn5))

newimpact = []

nI = 4
nH = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nbins = 50
for z in thresholds:
    for i in range(reps):
        #lesions
        #f1 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_LW_40_"+str(i)+".npy"))
        #f2 = 1 - (np.load("Eduardo/Data3/lesions_MCLW5_MC_40_"+str(i)+".npy"))

        #variance
        nn = np.load("Eduardo/Data3/state_MCLW5_LW_"+str(i)+".npy")
        nn = np.var(nn[:,nI:nI+nH],axis=0)
        max = np.max(nn)
        f1 = nn/max

        nn = np.load("Eduardo/Data3/state_MCLW5_MC_"+str(i)+".npy")
        nn = np.var(nn[:,nI:nI+nH],axis=0)
        max = np.max(nn)
        f2 = nn/max

        #print('before',len(f1))
        for p, (x,y) in enumerate(zip(f1,f2)):
            if x < z and y < z:
                #print(x)
                #indx = np.where(f1 == x)
                #print(indx)
                f1[p] = 200
                f2[p] = 100

        impact5[i] = (f1-f2)
        imp = impact5[i]
        imp = imp[imp != 100]
        newimpact.append(imp)

    flat_list = [item for sublist in newimpact for item in sublist]


    #ys = np.random.normal(loc=10, scale=10, size=2000)

    hist, bins = np.histogram(flat_list, bins=nbins)
    xs = (bins[:-1] + bins[1:])/2

    ax.bar(xs, hist, zs=z, zdir='y', color=cmap(hist/hist.max()),ec=cmap(hist/hist.max()), alpha=0.8)

ax.set_xlabel('Distance')
ax.set_ylabel('Threshold')
ax.set_zlabel('# of neurons')
ax.set_title("Distance of Variance: MCLW")
plt.show()







"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def triangulate_histogtam(x, y, z):

    if len(x)  != len(y) != len(z) :
        raise ValueError("The  lists x, y, z, must have the same length")
    n = len(x)
    if n % 2 :
        raise ValueError("The length of lists x, y, z must be an even number") 
    pts3d = np.vstack((x, y, z)).T
    pts3dp = np.array([[x[2*k+1], y[2*k+1], 0] for k in range(1, n//2-1)])
    pts3d = np.vstack((pts3d, pts3dp))
    #triangulate the histogram bars:
    tri = [[0,1,2], [0,2,n]]
    for k, i  in zip(list(range(n, n-3+n//2)), list(range(3, n-4, 2))):
        tri.extend([[k, i, i+1], [k, i+1, k+1]])
    tri.extend([[n-3+n//2, n-3, n-2], [n-3+n//2, n-2, n-1]])      
    return pts3d, np.array(tri)

# data
np.random.seed(123)
df = pd.DataFrame(np.random.normal(50, 5, size=(300, 4)), columns=list('ABCD'))

# plotly setup
fig = go.Figure()

# data binning and traces
bins = 10

bar_color = ['#e763fa', '#ab63fa', '#636efa', '#00cc96']
for m, col in enumerate(df.columns):
    a0=np.histogram(df[col], bins=bins, density=False)[0].tolist()
    a0=np.repeat(a0,2).tolist()
    a0.insert(0,0)
    a0.pop()
    a0[-1]=0
    a1=np.histogram(df[col], bins=bins-1, density=False)[1].tolist()
    a1=np.repeat(a1,2)

    verts, tri = triangulate_histogtam([m]*len(a0), a1, a0)
    x, y, z = verts.T
    I, J, K = tri.T
    fig.add_traces(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=bar_color[m], opacity=0.7))

fig.update_layout(width=700, height=700, scene_camera_eye_z=0.8)

"""