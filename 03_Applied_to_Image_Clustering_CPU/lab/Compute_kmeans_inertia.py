def Compute_kmeans_inertia(resultsDict, FSWRITE = False, gpu_available = False):
# Measure inertia of kmeans model for a variety of values of cluster number n
    km_list = list()
    data = resultsDict['PCA_fit_transform']
    #data = resultsDict['NP_images_STD']
    N_clusters = len(resultsDict['imagesFilenameList'])
    if gpu_available:
        print('Running Compute_kmeans_inertia on GPU: ')         
        with gpu_context():           
            for clust in range(1,N_clusters):
                km = KMeans(n_clusters=clust, init='random', random_state=42)
                km = km.fit(data)
                km_list.append({'clusters': clust, 
                                          'inertia': km.inertia_,
                                          'model': km})        
    # It is possible to specify to make the computations on CPU
    else:
        print('Running Compute_kmeans_inertia on local CPU: ')         
        for clust in range(1,N_clusters):
            km = KMeans(n_clusters=clust, init='random', random_state=42)
            km = km.fit(data)
            km_list.append({'clusters': clust, 
                                      'inertia': km.inertia_,
                                      'model': km})        
    resultsDict['km'] = km
    resultsDict['km_list'] = km_list
    if FSWRITE == True:
        write_results(resultsDict)
    return resultsDict
