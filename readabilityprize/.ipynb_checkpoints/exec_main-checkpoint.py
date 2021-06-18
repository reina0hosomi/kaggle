import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NLP_funcs as nlp

if __name__ == "__main__":
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    ###
    seri = train['excerpt']
    y = train['target']
    ins = nlp.NLP_Embedding(train_seri=seri,train_y = y)
    ins.trainModel()
    ##predict
    ts_X = test['excerpt']
    model = ins.loadPredictModel()
    prediction = ins.predict(test_x = ts_X,model=model)
    test['target'] = prediction
    test.to_csv("submission.csv",index=False)
    print("finish")