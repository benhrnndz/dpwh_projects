import pandas as pd
df = pd.read_csv("dpwh_flood_correlation_by_district.csv")
print(df.shape)         # (122, 30)
print(df.isnull().sum()) # All zeros
print(df.dtypes)