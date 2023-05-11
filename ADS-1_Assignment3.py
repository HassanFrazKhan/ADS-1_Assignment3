import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skdat
import seaborn as sns





df_co2 = pd.read_csv("World_population_rate_1.csv")
print(df_co2.describe())

df_co3 = df_co2[["1970", "1980", "1990", "2000", "2010", "2015"]]
print(df_co3.describe())


#transpose the data
print(df_co3.describe().T)





# define centres of three clusters
centres = [[-1., 0.], [1., -0.5], [0., 1.]]

# use make_blobs function to create dataset. 
# Points are normal distributed around the centres.
xy, nclust = skdat.make_blobs(1000, centers=centres, cluster_std=0.3)

for i in range(20):
    print(xy[i], nclust[i])
    

x = xy[:,0] # extract x and y vectors
y = xy[:,1]
print(centres)
xcent=[]
ycent=[]
for i in range(len(centres)):
    xcent.append(centres[i][0])
    ycent.append(centres[i][1])
print()
print(xcent)
print(ycent)




import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

corr_matrix = df.corr()
print(corr_matrix)




import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_ex = df_co3[["1990", "2015"]]  # extract the two columns for clustering


df_ex = df_ex.dropna()  # entries with one nan are useless
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])

# reset_index() moved the old index into column index
# remove before clustering
df_ex = df_ex.drop("index", axis=1)
print(df_ex.iloc[0:15])


corr_matrix = df_co3.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()



df_wp= pd.read_csv("World_population_rate_1.csv")
df_work = pd.read_csv("Age_dependency_ratio_1.csv")

print(df_wp.describe())
print(df_work.describe())
# drop rows with nan's in 2020
df_wp = df_wp[df_wp["2020"].notna()]
print(df_wp.describe())

# alternative way of targetting one or more columns
df_work = df_work.dropna(subset=["2020"])
print(df_work.describe)
df_agr2020 = df_wp[["Country Name", "Country Code", "2020"]].copy()
df_for2020 = df_work[["Country Name", "Country Code", "2020"]].copy()
print(df_agr2020.describe())
print(df_for2020.describe())
df_2020 = pd.merge(df_agr2020, df_for2020, on="Country Name", how="outer")
print(df_2020.describe())


print(df_2020.describe())
df_2020 = df_2020.dropna()    # entries with one datum or less are useless.
print()
print(df_2020.describe())

# rename columns
df_2020 = df_2020.rename(columns={"2020_x":"World Population Growth", "2020_y":"Working Population Proportion"})
pd.plotting.scatter_matrix(df_2020, figsize=(12, 12), s=5, alpha=0.8)




pd.plotting.scatter_matrix(df_co3, figsize=(12, 12), s=5, alpha=0.8)
plt.show()



#################
#################
#################


# cluster by cluster
plt.figure(figsize=(8.0, 8.0))

cm = plt.cm.get_cmap('tab10')
plt.scatter(x, y, 10, nclust, marker="o", cmap=cm)
plt.scatter(xcent, ycent, 45, "k", marker="d")
plt.xlabel("x")
plt.ylabel("y")
plt.show()





# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet

ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(xy)     # fit done on x,y pairs

labels = kmeans.labels_
# print(labels)    # labels is the number of the associated clusters of (x,y) points
# for i in range(50): 
#    print(xy[i], labels[i])
    
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(xy, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",         "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
       
for l in range(ncluster):     # loop over the different labels
    plt.plot(x[labels==l], y[labels==l], "o", markersize=4, color=col[l])
    
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("x")
plt.ylabel("y")
plt.show()




print(kmeans.predict([[0.5, 0.5]]))

ncluster = 3

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(xy)     # fit done on x,y pairs

labels = kmeans.labels_
    
# extract the estimated cluster centres
cen = kmeans.cluster_centers_

# calculate the silhoutte score
print(skmet.silhouette_score(xy, labels))




##################
##################
##################


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_function(x, a, b):
    """Linear function for fitting.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The input x-values.
    a : float
        The slope of the line.
    b : float
        The y-intercept of the line.
        
    Returns:
    --------
    numpy.ndarray
        The predicted y-values for the given x-values.
    """
    return a * x + b

# Load data from CSV file
data = np.genfromtxt('World_population_rate_2.csv', delimiter=',', skip_header=1, usecols=(1, 6), missing_values='', filling_values=np.nan)

x = data[:, 0]  # First column as x-vector
y = data[:, 1]  # Second column as y-vector

# Fit linear regression using curve_fit
mask = np.logical_and(np.isfinite(x), np.isfinite(y))
popt, pcov = curve_fit(linear_function, x[mask], y[mask])

# Extract fitted parameters
a_fit, b_fit = popt

# Create predicted y-values using fitted parameters
y_pred = linear_function(x, a_fit, b_fit)

# Plot the original data and the fitted line
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, 'r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the fitted parameters
print("Fitted Parameters:")
print("a =", a_fit)
print("b =", b_fit)
