Code for a lame contest hosted on CrowdAnalytix : https://www.crowdanalytix.com/contests/gamma-log-facies-type-prediction
Task: Multiclass classification. Predicting the type of rock formation (facies) based on gamma log values.

There was a glaringly obvious leak in the data -- repeating pattern of digits after the decimal point. 

Single LGB model 
CV - 0.9927, LB - 0.9935

Its possible to reach 0.995 on LB by saving oof predicted class labels, & feeding the predicted labels for current row, prev 5 rows & next 5 rows as features into a 2nd level LGB model 
