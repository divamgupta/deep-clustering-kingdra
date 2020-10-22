from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from kingdra_cluster.kingdra_cluster import KingdraCluster
import numpy as np 
from kingdra_cluster.unsup_metrics import ACC
import sys
import json




config = json.loads( open(sys.argv[1]).read())

gt = np.load (config['ground_truth_path'])
X = np.load(config['training_path'])

del config['ground_truth_path']
del config['training_path']


m = KingdraCluster(  **config  )


def callback( it , y_pred_e ,models  ):
	print("Clustering accuracy at  iteration " , it  , ACC( gt  ,  y_pred_e ) )

m.fit(X , callback=callback )

preds_2 = m.predict(X )



print("Clustering Accuracy: " , ACC( gt  ,  preds_2 ) )