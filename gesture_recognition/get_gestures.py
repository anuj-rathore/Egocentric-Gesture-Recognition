import subprocess, os

def assure_path_exists(path):
    #make dir if doesnt exist
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
                os.makedirs(dir)

#Path to videos with trailing slash
videosPath="/home/anuj/alpha2/"

subject_list=[name for name in os.listdir(videosPath)]
gesture_list=[name for name in os.listdir(videosPath+subject_list[0])]

#Path to DenseTrackse trajectories binary
dtbin="/home/anuj/dense_trajectory_release_v1.2/release/DenseTrack"

#Path where the DT features need to be stored, with trailing slash
DTPath="/home/anuj/DTfeatures/"
#read from files and create feature wise vetors..

import numpy as np
import os

TRAJ_DIM=30
HOG_DIM=96
HOF_DIM=108
MBHX_DIM=96
MBHY_DIM=96


totalvids=40
gesture1=20

traj_start =10
hog_start = traj_start + TRAJ_DIM
hof_start = hog_start + HOG_DIM
mbhx_start = hof_start + HOF_DIM
mbhy_start = mbhx_start + MBHX_DIM
mbhy_end = mbhy_start + MBHY_DIM

# Parses a video's IDTF file and returns a list of IDTF of that video
def extract_IDTF(vname):
    lines = []
    with open(vname) as f:
         lines= f.readlines()
    
    trajs=[]
    hogs=[]
    hofs=[]
    mbhxs=[]
    mbhys=[]
    
    for line in lines:
        ll = line.strip().split()
        traj = [float(l) for l in ll[traj_start:hog_start]]
        hog = [float(l) for l in ll[hog_start:hof_start]]
        hof = [float(l) for l in ll[hof_start:mbhx_start]]
        mbhx = [float(l) for l in ll[mbhx_start:mbhy_start]]
        mbhy = [float(l) for l in ll[mbhy_start:mbhy_end]]
    
            
        trajs.append(np.array(traj))
        hogs.append(np.array(hog))
        hofs.append(np.array(hof))
        mbhxs.append(np.array(mbhx))
        mbhys.append(np.array(mbhy))
    print (vname,"lines", len(lines))    
    return (trajs,hogs,hofs,mbhxs,mbhys)



alltrajs=[]
allhogs=[]
allhofs=[]
allmbhxs=[]
allmbhys=[]
labels=[]

videoFeatures=[]

#Iterate over videos
allvids=[]
labels=[]
l=0
sampled_vids=[]

for subject in subject_list:
    l=0
    for gesturename in gesture_list:
        videos_under_gesture=os.listdir(DTPath+subject+"/"+gesturename)
        #chose 10% of all vids randomly to generate codebooks
        sampled_vid_gest=np.random.choice(videos_under_gesture,int(3) , replace=False)
        labels+=[l]*len(videos_under_gesture)
        sampled_vids+=[DTPath+subject+"/"+gesturename+"/"+vname for vname in sampled_vid_gest.tolist()]
        allvids+=[DTPath+subject+"/"+gesturename+"/"+vname for vname in videos_under_gesture]
        l+=1

print ("Sampling the foll vids for generating codebooks...")
for v in range(len(sampled_vids)):
    traj,hog,hof,mbhx,mbhy=extract_IDTF(sampled_vids[v])
    alltrajs+=traj
    allhogs+=hog
    allhofs+=hof
    allmbhxs+=mbhx
    allmbhys+=mbhy
                   
print (len(alltrajs))



alltrajs = np.array(alltrajs)
allhogs = np.array(allhogs)
allhofs = np.array(allhofs)
allmbhxs = np.array(allmbhxs)
allmbhys = np.array(allmbhys)
labels=np.array(labels)

labels=labels.astype('int')

from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Perform k-means clustering & generate codebooks
k=500
traj_code_book, variance = kmeans(alltrajs, k)
hogs_code_book, variance = kmeans(allhogs, k)
hofs_code_book, variance = kmeans(allhofs, k)
mbhxs_code_book, variance = kmeans(allmbhxs, k)
mbhys_code_book, variance = kmeans(allmbhys, k)

codebook=[traj_code_book,hogs_code_book,hofs_code_book,mbhxs_code_book,mbhys_code_book]

def powernorm(hist):
     hist = np.sign(hist) * (np.abs(hist) ** 0.5)
     return hist



#Calculate of BOW per Video
#input IDFT of Video [(X,96), (X,100), (X,96), (X,100)]
#output np.array([(500,1),(500,1),(500,1),(500,1),(500,1)])
def computeBOW(videoFeat):
    BOW=[]
    print ("video feat len",len(videoFeat))
    for descrip,cbook in zip(videoFeat,codebook):
        #descrip (X,dim), #cbook(500,dim)
        hist,dist=vq(descrip,cbook)
        histo,bins= np.histogram(hist,np.arange(500), density=True)
        BOW.append(powernorm(histo))
    BOW=np.array(BOW)
    print ("BOW", BOW.shape)
    return BOW

# make vectors for training
features=[] #(numvideos,500, 4)
print (labels)
for vname in allvids:
    videoFeat=extract_IDTF(vname)
    features.append(computeBOW(videoFeat))
    
features=np.array(features)
labels=np.array(labels)
print ("")
print ("All Features Shape=",(features.shape))




from sklearn.cross_validation import train_test_split

#Split into test and train
x,des,dims=features.shape
features=features.reshape((x,des*dims))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=7)
print (features.shape, des,dims)

# Train the Linear SVM
from sklearn.svm import SVC
print (X_train.shape, X_test.shape)
clf = SVC(kernel='linear').fit(X_train, y_train)
print ("linearsvm", clf.score(X_test, y_test))


# Save the SVM
from sklearn.externals import joblib
joblib.dump((clf,codebook), "classifier.pkl", compress=3)
