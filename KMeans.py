import random
import pandas as pd

class KMeans:
    def __init__(self,data,numClusters,numIterations,epsilon):
        self.data = data
        self.numClusters = numClusters
        self.numIterations = numIterations
        self.epsilon = epsilon
        self.centroids = None
        self.clusters = None

# ****** set

    def setData(self,data):
        self.data = data

    def setNumClusters(self,numClusters):
        self.numClusters = numClusters

    def setEpsilon(self,epsilon):
        self.epsilon = epsilon

    def setCentroids(self,centroids):
        self.centroids = centroids

    def resetClusters(self):
        features = list(self.data.columns)
        features.append('cluster')
        features.append('distance')

        self.clusters = pd.DataFrame(columns=features)

# ****** get

    def getData(self):
        return self.data

    def getNumClusters(self):
        return self.numClusters

    def getEpsilons(self):
        return self.epsilon

    def getCentroids(self):
        return self.centroids

# ******

    def initializeCentroids(self):
        centroids = list()
        indices = random.sample(range(0,len(self.data)),self.numClusters)
        for i in indices:
            centroids.append(self.data.iloc[i, :])
        self.centroids = centroids

# ******

    def computeDistances(self):
        epsCheck = 0 # initialize epsilon check value
        for index, row in self.data.iterrows(): # iterate through each data row
            distance = list()
            i = 0
            for centroid in self.centroids: # iterate through each centroid
                d = list()
                for k in range(len(row)): # iterate through each data feature
                    d.append((row.iloc[k] - centroid[k]) ** 2) # calculate distance from each feature to centroid

                dist = sum(d) # sum the distances for the row

                distance.append([dist,i]) # append the distances/clustersIDs to a list
                i += 1

            c = sorted(distance)[0] # sort the distances and get the lowest distance
            row['cluster'] = c[1] # add cluster ID for lowest distance to data row
            row['distance'] = c[0] # add lowest distance to data row
            self.clusters = self.clusters.append(row) # add data row to self.clusters

            if c[0] >= self.epsilon: # check if the distance is greater than epsilon
                epsCheck = 1

        return epsCheck

# ******

    def computeCentroids(self):
        centroids = list()

        for i in range(self.numClusters): # for Number of clusters
            c = self.clusters.loc[self.clusters['cluster'] == i] # get data frame for individual clusters
            featValues = c.drop(columns=['cluster','distance']) # remove 'cluster' and 'distance'

            centroids.append(featValues.mean()) # get them mean column values for each feature in the cluster and append
        self.centroids = centroids

    # ******

    def fixEmptyClusters(self):
        clusterCounts = self.clusters['cluster'].value_counts()  # Number of points in each cluster

        emptyClusters = [x for x in range(0, self.numClusters) if
                         x not in clusterCounts]  # Find clusters with no points

        if len(emptyClusters) == 0:
            return

        indexedClusters = self.clusters.copy()
        indexedClusters['index'] = range(0, len(indexedClusters))  # Copy of clusters with an index column

        sortedClusters = indexedClusters.sort_values(by=['distance'],
                                                     ascending=False)  # Sorted by worst distance to centroid

        k = 0
        while len(emptyClusters) > 0 and k < len(sortedClusters):
            candidatePoint = sortedClusters.iloc[k]  # Try to make worst distance a new centroid

            if (clusterCounts[
                candidatePoint['cluster']] > 1):  # Make sure moving the point won't make a new empty clusters
                clusterCounts[candidatePoint['cluster']] -= 1

                clusterToFill = emptyClusters.pop(0)
                self.centroids[clusterToFill] = candidatePoint.drop(
                    columns=['cluster', 'distance', 'index'])  # Create new centroid with values

                # Modify row in clusters to match new values
                self.clusters.at[candidatePoint['index'], 'cluster'] = clusterToFill
                self.clusters.at[candidatePoint['index'], 'distance'] = 0

            k += 1



    def runClustering(self):
        self.initializeCentroids()

        for i in range(self.numIterations):
            self.resetClusters()
            epsCheck = self.computeDistances()
            self.fixEmptyClusters()

            if epsCheck == 0:
                print('All distances less than epsilon (%d)' % self.epsilon)
                break

            self.computeCentroids()
            print('Completed iteration %d' % i)

# ******

if __name__ == '__main__':

    data = pd.read_csv('iris.csv') # read the data

    labels = data['class'] # get the class labels

    data = data.drop(columns=['class']) # get the clustering data

    km = KMeans(data,3,10,1)  # create instance of KMeans(data,numClusters,numIterations,epsilon)
    km.runClustering() # run the clustering algorithm

    clusters = km.clusters
    clusters['class'] = labels # add the class labels to the cluster data frame

    print('\n\n** Printing Cluster Data Sheet **\n\n', clusters)
    clusters.to_csv('clusters.csv') # output clusters and labels to .csv