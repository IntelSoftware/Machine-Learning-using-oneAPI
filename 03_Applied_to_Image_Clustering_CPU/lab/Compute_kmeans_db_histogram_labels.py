
def Compute_kmeans_db_histogram_labels(resultsDict, knee = 5, EPS = 350, n = 3, FSWRITE = False, gpu_available = False):
    imageClusters = knee
    PCA_images = resultsDict['PCA_fit_transform']
    if gpu_available:
        print('Running Compute_kmeans_histogram_labels on GPU: ')         
        with gpu_context():           
            k_means = KMeans(n_clusters = imageClusters, init='random')
            PCA_fit_transform = resultsDict['PCA_fit_transform']
            db = DBSCAN(eps=EPS, min_samples=n).fit(PCA_images)
            #PCA_fit_transform = resultsDict['NP_images_STD']
            k_means.fit(PCA_fit_transform)       

    # It is possible to specify to make the computations on CPU
    else:
        print('Running Compute_kmeans_histogram_labels on local CPU: ')         
        k_means = KMeans(n_clusters = imageClusters, init='random')
        PCA_fit_transform = resultsDict['PCA_fit_transform']
        db = DBSCAN(eps=EPS, min_samples=n).fit(PCA_images)
        #PCA_fit_transform = resultsDict['NP_images_STD']
        k_means.fit(PCA_fit_transform)

    imageClusters_db = len(np.unique(db.labels_))
    counts_db, bins_db  = np.histogram(db.labels_, bins=imageClusters_db)
    counts, bins =np.histogram(k_means.labels_, bins=imageClusters)
    resultsDict['counts_db'] = counts_db
    resultsDict['bins_db'] = bins_db
    resultsDict['counts'] = counts
    resultsDict['bins'] = bins
    resultsDict['model_db'] = db
    resultsDict['model'] = k_means  
    resultsDict['imageClusters'] = imageClusters
    resultsDict['imageClusters_db'] = imageClusters_db
    if FSWRITE == True:
        write_results(resultsDict)
    return resultsDict
