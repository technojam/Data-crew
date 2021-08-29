import pandas as pd
penguins = pd.read_csv('clean_dataset.csv')

# Ordinal feature encoding

df = penguins.copy()
target = 'EstimatedDayofOvulation'
encode = ['LengthofCycle','LengthofLutealPhase']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper =  {17:17, 15:15, 16:16, 14:14, 18:18, 12:12, 19:19, 11:11, 13:13, 27:27, 22:22,  8:8, 20:20, 21:21, 23:23, 10:10, 26:26,24:24, 29:29,  9:9, 25:25, 28:28,  6:6}
def target_encode(val):
    return target_mapper[val]

df['EstimatedDayofOvulation'] = df['EstimatedDayofOvulation'].apply(target_encode)

# Separating X and y
X = df.drop('EstimatedDayofOvulation', axis=1)
Y = df['EstimatedDayofOvulation']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('cycle_model.pkl', 'wb'))
