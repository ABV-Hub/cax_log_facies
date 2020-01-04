import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold as GKF
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import log_loss as ll, accuracy_score as asc
import gc

pd.set_option('display.max_columns', 50)

df_train = pd.read_csv('../input/CAX_LogFacies_Train_File.csv')
df_test = pd.read_csv('../input/CAX_LogFacies_Test_File.csv')

df_train.well_id = df_train.well_id.astype('int16')
df_train.label = df_train.label.astype('int8')
df_train.row_id = df_train.row_id.astype('int16')

df_test.well_id = df_test.well_id.astype('int16')
df_test.row_id = df_test.row_id.astype('int16')

Y = df_train['label'].values
unique_id = df_test['unique_id'].values

df_train.drop('label', axis=1, inplace=True)
df_test.drop('unique_id', axis=1, inplace=True)

df_full = pd.concat([df_train,df_test], axis=0).reset_index(drop=True)

#for df in [df_train, df_test]:
for df in [df_full]:
    #last 4 digits after decimal
    df['digits'] = df['GR'].map(lambda x: str(x).split(".")[1][2:6])
    df['digits'] = df['digits'].replace('','0')
    df['digits'] = df['digits'].astype('int')
    
    #forward & backward differences
    df['diff_1'] = df.groupby(['well_id'])['GR'].diff(1)    
    df['diff_m1'] = df.groupby(['well_id'])['GR'].diff(-1)
    df['diff_2'] = df.groupby(['well_id'])['GR'].diff(2)
    df['diff_m2'] = df.groupby(['well_id'])['GR'].diff(-2)
    df['diff_3'] = df.groupby(['well_id'])['GR'].diff(3)
    df['diff_m3'] = df.groupby(['well_id'])['GR'].diff(-3)
    df['diff_5'] = df.groupby(['well_id'])['GR'].diff(5)
    df['diff_m5'] = df.groupby(['well_id'])['GR'].diff(-5)
    df['diff_10'] = df.groupby(['well_id'])['GR'].diff(10)
    df['diff_m10'] = df.groupby(['well_id'])['GR'].diff(-10)
    df['diff_20'] = df.groupby(['well_id'])['GR'].diff(20)
    df['diff_m20'] = df.groupby(['well_id'])['GR'].diff(-20)
    df['diff_50'] = df.groupby(['well_id'])['GR'].diff(50)
    df['diff_m50'] = df.groupby(['well_id'])['GR'].diff(-50)
    
    #forward & backward shifts
    df['shift_1'] = df.groupby(['well_id'])['GR'].shift(1)
    df['shift_2'] = df.groupby(['well_id'])['GR'].shift(2)
    df['shift_3'] = df.groupby(['well_id'])['GR'].shift(3)
    df['shift_5'] = df.groupby(['well_id'])['GR'].shift(5)
    df['shift_10'] = df.groupby(['well_id'])['GR'].shift(10)
    df['shift_20'] = df.groupby(['well_id'])['GR'].shift(20)
    df['shift_m1'] = df.groupby(['well_id'])['GR'].shift(-1)
    df['shift_m2'] = df.groupby(['well_id'])['GR'].shift(-2)
    df['shift_m3'] = df.groupby(['well_id'])['GR'].shift(-3)
    df['shift_m5'] = df.groupby(['well_id'])['GR'].shift(-5)
    df['shift_m10'] = df.groupby(['well_id'])['GR'].shift(-10)
    df['shift_m20'] = df.groupby(['well_id'])['GR'].shift(-20)
    df['shift_50'] = df.groupby(['well_id'])['GR'].shift(50)
    df['shift_m50'] = df.groupby(['well_id'])['GR'].shift(-50)
    
    #simple moving avgs of GR, shifts
    df['sma_5'] = df.groupby(['well_id'])['GR'].rolling(window=5).mean().reset_index(0,drop=True)
    df['sma_10'] = df.groupby(['well_id'])['GR'].rolling(window=10).mean().reset_index(0,drop=True)
    df['sma_20'] = df.groupby(['well_id'])['GR'].rolling(window=20).mean().reset_index(0,drop=True)
    df['sma_50'] = df.groupby(['well_id'])['GR'].rolling(window=50).mean().reset_index(0,drop=True)
    df['sma_m5'] = df.groupby(['well_id'])['shift_m5'].rolling(window=5).mean().reset_index(0,drop=True)
    df['sma_m10'] = df.groupby(['well_id'])['shift_m10'].rolling(window=10).mean().reset_index(0,drop=True)
    df['sma_m20'] = df.groupby(['well_id'])['shift_m20'].rolling(window=20).mean().reset_index(0,drop=True)
    df['sma_m50'] = df.groupby(['well_id'])['shift_m50'].rolling(window=50).mean().reset_index(0,drop=True)
    
    #some ratios from diff features & sma features
    df['ratio_1'] = df['diff_1']/df['diff_m1']
    df['ratio_3'] = df['diff_3']/df['diff_m3']
    df['ratio_5'] = df['diff_5']/df['diff_m5']
    df['ratio_10'] = df['diff_10']/df['diff_m10']
    df['ratio_20'] = df['diff_20']/df['diff_m20']
    df['ratio_50'] = df['diff_50']/df['diff_m50']
    
    df['ratio_sma_5'] = df['sma_5']/df['sma_m5']
    df['ratio_sma_10'] = df['sma_10']/df['sma_m10']
    df['ratio_sma_20'] = df['sma_20']/df['sma_m20']
    df['ratio_sma_50'] = df['sma_50']/df['sma_m50']
    
    df['ratio_sma_5_10'] = df['sma_5']/df['sma_10']
    df['ratio_sma_10_20'] = df['sma_10']/df['sma_20']
    df['ratio_sma_m5_m10'] = df['sma_m5']/df['sma_m10']
    df['ratio_sma_m10_m20'] = df['sma_m10']/df['sma_m20']
    df['ratio_sma_m20_m50'] = df['sma_m20']/df['sma_m50']
    
    #moving stds, & ratios from these
    df['smstd_5'] = df.groupby(['well_id'])['GR'].rolling(window=5).std().reset_index(0,drop=True)
    df['smstd_10'] = df.groupby(['well_id'])['GR'].rolling(window=10).std().reset_index(0,drop=True)
    df['smstd_20'] = df.groupby(['well_id'])['GR'].rolling(window=20).std().reset_index(0,drop=True)
    df['smstd_50'] = df.groupby(['well_id'])['GR'].rolling(window=50).std().reset_index(0,drop=True)
    df['smstd_m5'] = df.groupby(['well_id'])['shift_m5'].rolling(window=5).std().reset_index(0,drop=True)
    df['smstd_m10'] = df.groupby(['well_id'])['shift_m10'].rolling(window=10).std().reset_index(0,drop=True)
    df['smstd_m20'] = df.groupby(['well_id'])['shift_m20'].rolling(window=20).std().reset_index(0,drop=True)
    df['smstd_m50'] = df.groupby(['well_id'])['shift_m50'].rolling(window=50).std().reset_index(0,drop=True)
    
    df['ratio_smstd_5'] = df['smstd_5']/df['smstd_m5']
    df['ratio_smstd_10'] = df['smstd_10']/df['smstd_m10']
    df['ratio_smstd_20'] = df['smstd_20']/df['smstd_m20']
    df['ratio_smstd_50'] = df['smstd_50']/df['smstd_m50']
    
    df['ratio_smstd_5_10'] = df['smstd_5']/df['smstd_10']
    df['ratio_smstd_10_20'] = df['smstd_10']/df['smstd_20']
    df['ratio_smstd_m5_m10'] = df['smstd_m5']/df['smstd_m10']
    df['ratio_smstd_m10_m20'] = df['smstd_m10']/df['smstd_m20']
    df['ratio_smstd_m20_m50'] = df['smstd_m20']/df['smstd_m50']
    
    #forward & bckward diffs of digits
    df['diff_digits1'] = df.groupby(['well_id'])['digits'].diff(1)
    df['diff_digitsm1'] = df.groupby(['well_id'])['digits'].diff(-1)
    df['diff_digits2'] = df.groupby(['well_id'])['digits'].diff(2)
    df['diff_digitsm2'] = df.groupby(['well_id'])['digits'].diff(-2)
    df['diff_digits3'] = df.groupby(['well_id'])['digits'].diff(3)
    df['diff_digitsm3'] = df.groupby(['well_id'])['digits'].diff(-3)
    df['diff_digits5'] = df.groupby(['well_id'])['digits'].diff(5)
    df['diff_digitsm5'] = df.groupby(['well_id'])['digits'].diff(-5)
    df['diff_digits10'] = df.groupby(['well_id'])['digits'].diff(10)
    df['diff_digitsm10'] = df.groupby(['well_id'])['digits'].diff(-10)
    df['diff_digits20'] = df.groupby(['well_id'])['digits'].diff(20)
    df['diff_digitsm20'] = df.groupby(['well_id'])['digits'].diff(-20)
    
    df['shift_digitsm5'] = df.groupby(['well_id'])['digits'].shift(-5)
    df['shift_digitsm10'] = df.groupby(['well_id'])['digits'].shift(-10)
    df['shift_digitsm20'] = df.groupby(['well_id'])['digits'].shift(-20)
    
    #moving avgs & stds of digits, & ratios from them
    df['sma_digits5'] = df.groupby(['well_id'])['digits'].rolling(window=5).mean().reset_index(0,drop=True)
    df['sma_digits10'] = df.groupby(['well_id'])['digits'].rolling(window=10).mean().reset_index(0,drop=True)
    df['sma_digits20'] = df.groupby(['well_id'])['digits'].rolling(window=20).mean().reset_index(0,drop=True)
    df['sma_digitsm5'] = df.groupby(['well_id'])['shift_digitsm5'].rolling(window=5).mean().reset_index(0,drop=True)
    df['sma_digitsm10'] = df.groupby(['well_id'])['shift_digitsm10'].rolling(window=10).mean().reset_index(0,drop=True)
    df['sma_digitsm20'] = df.groupby(['well_id'])['shift_digitsm20'].rolling(window=20).mean().reset_index(0,drop=True)
    
    df['ratio_sma_digits5'] = df['sma_digits5']/df['sma_digitsm5']
    df['ratio_sma_digits10'] = df['sma_digits10']/df['sma_digitsm10']
    df['ratio_sma_digits20'] = df['sma_digits20']/df['sma_digitsm20']
    
    df['ratio_sma_digits5_10'] = df['sma_digits5']/df['sma_digits10']
    df['ratio_sma_digits10_20'] = df['sma_digits10']/df['sma_digits20']
    df['ratio_sma_digitsm5_m10'] = df['sma_digitsm5']/df['sma_digitsm10']
    df['ratio_sma_digitsm10_m20'] = df['sma_digitsm10']/df['sma_digitsm20']
    
    df['smstd_digits5'] = df.groupby(['well_id'])['digits'].rolling(window=5).std().reset_index(0,drop=True)
    df['smstd_digits10'] = df.groupby(['well_id'])['digits'].rolling(window=10).std().reset_index(0,drop=True)
    df['smstd_digits20'] = df.groupby(['well_id'])['digits'].rolling(window=20).std().reset_index(0,drop=True)
    df['smstd_digitsm5'] = df.groupby(['well_id'])['shift_digitsm5'].rolling(window=5).std().reset_index(0,drop=True)
    df['smstd_digitsm10'] = df.groupby(['well_id'])['shift_digitsm10'].rolling(window=10).std().reset_index(0,drop=True)
    df['smstd_digitsm20'] = df.groupby(['well_id'])['shift_digitsm20'].rolling(window=20).std().reset_index(0,drop=True)
    
    df['ratio_smstd_digits5'] = df['smstd_digits5']/df['smstd_digitsm5']
    df['ratio_smstd_digits10'] = df['smstd_digits10']/df['smstd_digitsm10']
    df['ratio_smstd_digits20'] = df['smstd_digits20']/df['smstd_digitsm20']
    
    df['ratio_smstd_digits5_10'] = df['smstd_digits5']/df['smstd_digits10']
    df['ratio_smstd_digits10_20'] = df['smstd_digits10']/df['smstd_digits20']
    df['ratio_smstd_digitsm5_m10'] = df['smstd_digitsm5']/df['smstd_digitsm10']
    df['ratio_smstd_digitsm10_m20'] = df['smstd_digitsm10']/df['smstd_digitsm20']
    
    del df
    gc.collect()


#min & max rowid for digit pattern within each well    
g = df_full.groupby(['well_id','digits'])['row_id'].agg(['min','max']).reset_index()
g.columns = ['well_id','digits','digit_rowid_min','digit_rowid_max']

#digit pattern freq within each well    
g2 = df_full.groupby(['well_id','digits']).size().reset_index()
g2.columns = ['well_id','digits','digit_freq']

#cumulative count of digit pattern within each well    
g3 = df_full.groupby(['well_id','digits']).cumcount()
df_full['digit_cumcount'] = g3 + 1

df_full = df_full.merge(g, how='left')
df_full = df_full.merge(g2, how='left')
#some indication of how many more rows to follow with the current digit pattern
df_full['cumcount_freq_ratio'] = df_full['digit_cumcount']/df_full['digit_freq']
df_full['rowid_max_delta'] = df_full['digit_rowid_max'] - df_full['row_id']

#moving avgs & stds of digit freq
df_full['sma_freq_5'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=5).mean().reset_index(0,drop=True)
df_full['sma_freq_10'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=10).mean().reset_index(0,drop=True)
df_full['sma_freq_20'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=20).mean().reset_index(0,drop=True)
df_full['sma_freq_m5'] = df_full.groupby(['well_id'])['digit_freq'].shift(-5).rolling(window=5).mean().reset_index(0,drop=True)
df_full['sma_freq_m10'] = df_full.groupby(['well_id'])['digit_freq'].shift(-10).rolling(window=10).mean().reset_index(0,drop=True)
df_full['sma_freq_m20'] = df_full.groupby(['well_id'])['digit_freq'].shift(-20).rolling(window=20).mean().reset_index(0,drop=True)

df_full['smstd_freq_5'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=5).std().reset_index(0,drop=True)
df_full['smstd_freq_10'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=10).std().reset_index(0,drop=True)
df_full['smstd_freq_20'] = df_full.groupby(['well_id'])['digit_freq'].rolling(window=20).std().reset_index(0,drop=True)
df_full['smstd_freq_m5'] = df_full.groupby(['well_id'])['digit_freq'].shift(-5).rolling(window=5).std().reset_index(0,drop=True)
df_full['smstd_freq_m10'] = df_full.groupby(['well_id'])['digit_freq'].shift(-10).rolling(window=10).std().reset_index(0,drop=True)
df_full['smstd_freq_m20'] = df_full.groupby(['well_id'])['digit_freq'].shift(-20).rolling(window=20).std().reset_index(0,drop=True)

#forward & backward shifts of rowid_max_delta
df_full['rowid_max_delta_shift1'] = df_full['rowid_max_delta'].shift(1)
df_full['rowid_max_delta_shiftm1'] = df_full['rowid_max_delta'].shift(-1)
df_full['rowid_max_delta_shift3'] = df_full['rowid_max_delta'].shift(3)
df_full['rowid_max_delta_shiftm3'] = df_full['rowid_max_delta'].shift(-3)
df_full['rowid_max_delta_shift5'] = df_full['rowid_max_delta'].shift(5)
df_full['rowid_max_delta_shiftm5'] = df_full['rowid_max_delta'].shift(-5)
df_full['rowid_max_delta_shift7'] = df_full['rowid_max_delta'].shift(7)
df_full['rowid_max_delta_shiftm7'] = df_full['rowid_max_delta'].shift(-7)
df_full['rowid_max_delta_shift10'] = df_full['rowid_max_delta'].shift(10)
df_full['rowid_max_delta_shiftm10'] = df_full['rowid_max_delta'].shift(-10)

#some diff features from digit_freq & its moving avgs
df_full['freqdiff_5'] = df_full['digit_freq'] - df_full['sma_freq_5']
df_full['freqdiff_10'] = df_full['digit_freq'] - df_full['sma_freq_10']
df_full['freqdiff_20'] = df_full['digit_freq'] - df_full['sma_freq_20']
df_full['freqdiff_m5'] = df_full['digit_freq'] - df_full['sma_freq_m5']
df_full['freqdiff_m10'] = df_full['digit_freq'] - df_full['sma_freq_m10']
df_full['freqdiff_m20'] = df_full['digit_freq'] - df_full['sma_freq_m20']

print(df_full.shape)

float_cols = df_full.select_dtypes(include='float').columns.values
for c in float_cols:
    df_full[c] = df_full[c].astype('float32')
    
df_train = df_full[:len(df_train)]
df_test = df_full[len(df_train):]
print(df_train.shape, df_test.shape)

del df_full
gc.collect()

useless = ['well_id']
useful = [c for c in list(df_train) if c not in useless]
print(len(useful))


well_tr, well_val = tts(df_train.well_id.unique(), test_size=0.2, random_state=123)
print (len(well_tr))
print (len(well_val))

tr = df_train[df_train.well_id.isin(well_tr)]
val = df_train[df_train.well_id.isin(well_val)]

Y_tr = Y[tr.index.values]
Y_val = Y[val.index.values]

nrounds = 500
lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'multiclass',
    'num_class':5,
    'nthread': -1,
    'num_leaves': 2**10, 'learning_rate': 0.2, #0.2 
    'max_depth': 10,
    'max_bin': 1024,  
    'colsample_bytree': 0.35, 
    'metric': 'multi_error', #'multi_logloss'
    'min_child_weight': 2, 'min_child_samples': 10, 
    'reg_alpha': 0.05, 'reg_lambda': 0.05, 
    'bagging_fraction': 0.9, 
    'bagging_freq': 5}

ltr = lgb.Dataset(tr,Y_tr)
lval = lgb.Dataset(val,Y_val, reference= ltr)

gbdt = lgb.train(lgb_params, ltr, nrounds, valid_sets=lval,
                         verbose_eval=50,
                         early_stopping_rounds=50)  #50
bst=gbdt.best_iteration
pred=gbdt.predict(val, num_iteration=bst)

scr=asc(Y_val,np.argmax(pred, axis=1) )
if bst ==-1:
    bst = nrounds

del ltr
del lval

gc.collect()

print(scr)
print(bst)

pred_class = np.argmax(pred, axis=1)
print(pd.Series(pred_class).value_counts(True))
print(pd.Series(Y_val).value_counts(True))

ltrain=lgb.Dataset(df_train[useful],Y)

gbdt = lgb.train(lgb_params, ltrain, bst)
pred=gbdt.predict(df_test[useful])

pred_class_test = np.argmax(pred, axis=1)

submit = pd.DataFrame({'unique_id': unique_id, 'label': pred_class_test})

print(pd.Series(Y).value_counts(True))
print(submit['label'].value_counts(True))

submit.to_csv('../output/leaksub.csv', index=False)
   