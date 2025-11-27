from sklearn.preprocessing import StandardScaler

# select numeric columns
numeric_features = df[["year", "average_rating"]]

# create scaler
scaler = StandardScaler()

# fit + transform
scaled_numeric = scaler.fit_transform(numeric_features)

# convert back to DataFrame
scaled_numeric = pd.DataFrame(scaled_numeric, columns=["year", "average_rating"])
