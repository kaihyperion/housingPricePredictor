## Housing Price Predictor for California
Using housing dataset csv, I converted the csv to Dataframe and first examined the data.

Created appropriate visualization, then split the training and test dataset (stratified by income categories)

Added extra attribute by viewing attribute correlation matrix

used `SKlearn.impute.SimpleImputer` to fill in missing data points with median of the appropriate attributes.

Used `sklearn.preprocessing.OneHotEncoder` to create a one binary attribute of the "ocean_proximity" category

Create Transformation Pipelines using `sklearn.pipeline.Pipeline` to prepare the data

Trained and evaluated the model using multiple complex models: `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `SVM`

Fine tuned the model using `GridSearchCV` for `RandomForestRegressor`

then using grid search's best estimator, I've analyzed the importance scores on the corresponding attributes (features)

Finally, ran evaluation on the test set and resulted in 95% confidence interval of 

array([47242.58280073, 51096.10323509])
