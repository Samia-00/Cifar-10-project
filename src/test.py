# testing & submitting kaggle cifar-10 test images
import numpy as np
import pandas as pd
import os

#project module

from .. import config
from . import preprocess, my_model

#loading model
model=my_model.read_model()
label =[ "airplane", "automobile", "bird", "cat",
    "deer" ,"dog", "frog", "horse", "ship", "truck"]

#loading test data
result = []
for part in range(0,6):
    x_test = preprocess.get_test_data_by_part(part)

    #predicting results
    print("Predicting results")
    predictions = model.predict(x_test,
                               batch_size = config.batch_size,
                               verbose = 2)
    #print(predictions.shape)

    label_pred = np.argmax(predictions,axis = 1)
    #print(label_pred)

    result += label_pred.tolist()
    #print(result)

#submitting result into csv file

result = [label[i] for i in result]
#print(result)

submit_df = pd.DataFrame({"id": range(1, config.nb_test_samples + 1),
                   "label":result})
submit_df.to_csv(os.path.join(config.submission_path(), "baseline_sub.csv"),
                 header = True, index = False)
print("Submission file Saved successfully")

    