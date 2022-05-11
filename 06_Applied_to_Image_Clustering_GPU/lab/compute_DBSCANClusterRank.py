def compute_DBSCANClusterRank(n, EPS):
    d = {index-1:int(cnt) for index, cnt in enumerate(counts )}
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0, len(d)):
        idx = sorted_d[i][0]
        print('cluster = ', idx, ' occurs', int(sorted_d[i][1]), ' times')
    return db, counts, bins, sorted_d

n_components = 4

columns = ['PC{:0d}'.format(c) for c in range(n_components)]
data = pd.DataFrame(resultsDict['PCA_fit_transform'][:,:n_components], columns = columns)

columns.append('cluster')
data['cluster'] = resultsDict['model_db']['labels_'] 
data.head()
