def ComputePCA(resultsDict, n_components=3, gpu_available = False):
    """
    ComputePCA(NP_images, n_components=10 )
    
    Computes a specified number of principal components given and numpy array image.
    
    inputs:
        NP_images - a numpy array of image data - in this case - the array contains color information for every pixel in an image in a wide single row per image concatenated vertically
        
        This means the NP_images.shape first dimensions represents the number of images
        while the second dimension represents the number of features
        
        n_components - specifies the number of PC to compute
    return:
        data - a pandas dataframe of PCs
        PCA_images - the numpy array of PC's
    """       
    NP_images_STD = resultsDict['NP_images_STD']
    # It is possible to specify to make the computations on GPU
    if gpu_available:
        print('Running ComputePCA on GPU: ')         
        with gpu_context():           
            pca = PCA(n_components=n_components)
            PCA_fit_transform = pca.fit_transform(NP_images_STD)        

    # It is possible to specify to make the computations on CPU
    else:
        print('Running ComputePCA on local CPU: ')         
        pca = PCA(n_components=n_components)
        PCA_fit_transform = pca.fit_transform(NP_images_STD)        

    columns = ['PC{:02d}'.format(pc) for pc in range(0,n_components) ]

    resultsDict['pca'] = pca
    resultsDict['PCA_fit_transform'] = PCA_fit_transform 
    return resultsDict
