#%%
import pandas as pd

#%%
results = pd.read_csv('model/cv_results.csv')

# %%
results[['param_anomaly_algorithm','param_tsvd__n_components', 'mean_test_Ratio anomalies Overall', 'std_test_Ratio anomalies Overall', 'mean_test_Ratio anomalies Oncologie', 'std_test_Ratio anomalies Oncologie','mean_test_Ratio anomalies Ob/gyn', 'std_test_Ratio anomalies Ob/gyn','mean_test_Ratio anomalies Pédiatrie', 'std_test_Ratio anomalies Pédiatrie','mean_test_Ratio anomalies Néonatologie', 'std_test_Ratio anomalies Néonatologie']]

# %%
