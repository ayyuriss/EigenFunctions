import core.utils as U
import core.picklize as pkl
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from munkres import Munkres
from sklearn.neighbors import kneighbors_graph
import numpy as np
import scipy.sparse as SP

def build_laplacian(X,nearest):
    batch_size = X.shape[0]
    W = kneighbors_graph(X.reshape(batch_size,-1), nearest, mode='distance', include_self=True, n_jobs=2)
    sigma = np.median(np.max(W.toarray(), axis=1))
    W = (W+W.getH())/2
    W.data = np.exp(-W.data**2/sigma**2)
    L = SP.coo_matrix(np.diag(np.sum(W.toarray(),axis=0))-W.toarray())
    return L
    
def munkres_acc(Vects, true_labels,start=1, max_k = 25,n_clusters=None):
    n = len(np.unique(true_labels))
    assert n == np.max(true_labels)+1
    acc = np.zeros(max_k)
    m = Munkres()
    scaler = StandardScaler()
    Vec = scaler.fit_transform(Vects[:,start:max_k+start])
    if n_clusters is None:
        n_clusters = n
    for l in range(max_k):
        V = Vec[:,:l+1]
        V = scaler.fit_transform(V)
        V = (V.T/np.linalg.norm(V,axis=1)).T
        km = KMeans(n_clusters=n_clusters,n_init=30,tol=1e-8,max_iter=500)
        clusters = km.fit_predict(V)
        true_hot = np.eye(n)[true_labels]
        pred_hot = np.eye(n_clusters)[clusters]
        mat = pred_hot.T.dot(true_hot)
        permute = np.array(m.compute(-mat))[:,1]
        ac = np.mean(permute[clusters]==true_labels)
        print("Accuracy for 1...%i"%(l+1),ac,"NMI:",sklearn.metrics.normalized_mutual_info_score(clusters,true_labels))
        acc[l] = ac
    return acc

def corr_matrix(X,Y,tolx=1e-12,toly=1e-12):
    corr = -np.ones((X.shape[1],Y.shape[1]))
    sX = np.std(X,axis=0)
    sY = np.std(Y,axis=0)
    corr[np.logical_and(sY[:,None]<toly,sX[None,:]<tolx).T] = 1
    corr[np.logical_xor(sY[:,None]<toly,sX[None,:]<tolx).T] = 0
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if corr[i,j]==-1:
                mx = X[:,i]-np.mean(X[:,i])
                my = Y[:,j]-np.mean(Y[:,j])
                corr[i,j] = np.mean(mx*my)/(sX[i]*sY[j])
    return corr/(U.LA.norm(corr,axis=0)+U.EPS)



class DistanceLearner(object):
    
    def __init__(self, n_data,nearest,max_neighbr,name):
        
        self.D = SP.lil_matrix((n_data,n_data))
        self.near = nearest
        self.max = max_neighbr
        self.sig = 0
        self.name = name
    def submit(self,X,batch):
        W = kneighbors_graph(X.reshape(X.shape[0],-1), self.near, mode='distance', include_self=True, n_jobs=2)
        idx = W.nonzero()
        for i,j in zip(*idx):
            self.D[batch[i],batch[j]] = W[i,j]
        self.reduce()
        self.sig = np.mean(self.D[self.D.nonzero()].data[0])
        W =  self.D[batch][:,batch].copy()
        #W = W.maximum(W.getH())
        W = (W + W.getH())/2
        W.data = np.exp(-W.data**2/self.sig**2)
        return W
    def reduce(self):
        idx = self.D.nonzero()
        dic = dict()
        for i,j in zip(*idx):
            res = dic.get(i,([],0))
            x = res[0] + [j]
            y = res[1] + 1
            dic[i] = (x,y)
        for k,v in dic.items():
            if v[1]>self.max:
                vals = self.D[k,v[0]]
                idx = np.argsort(vals.toarray())[0]
                for i in idx[self.max:]:
                    self.D[k,v[0][i]] = 0
    def save(self,path):
        print("Saving Distance Learner")
        try:
            pkl.gpicklize(self.D,path+self.name+"D.pkl.gz")
        except:
            import core.console as C
            C.warning("Couldn't save " + path+self.name+"D.pkl.gz")

    def load(self,path):
        print("Loading Distance Learner")
        try :
            self.D = pkl.gdepicklize(path+self.name+"D.pkl.gz")
        except:
            import core.console as C
            C.warning("Couldn't load "+path+self.name+"D.pkl.gz")
                    
def munkres_test(Vects, true_labels,n_clusters=None):
    n_l = len(np.unique(true_labels))
    assert n_l == np.max(true_labels)+1
    m = Munkres()
    scaler = StandardScaler()
    V = scaler.fit_transform(Vects)
    #V = Vects
    if n_clusters is None:
        n_clusters = n_l
    print("Evaluating Clustering, clusters:",n_clusters,",vectors:",Vects.shape[1])
    V = (V.T/np.linalg.norm(V,axis=1)).T
    km = KMeans(n_clusters=n_clusters)
    km = KMeans(n_clusters=n_clusters)
    clusters = km.fit_predict(V)
    true_hot = np.eye(n_l)[true_labels]
    pred_hot = np.eye(n_clusters)[clusters]
    mat = pred_hot.T.dot(true_hot)
    permute = np.array(m.compute(-mat))[:,1]
    ac = np.mean(permute[clusters]==true_labels)
    nmi = sklearn.metrics.normalized_mutual_info_score(clusters,true_labels)
    print("Accuracy:",ac,"NMI:",nmi)
    return ac,nmi
