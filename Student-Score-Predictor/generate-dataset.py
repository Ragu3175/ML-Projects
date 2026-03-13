import pandas as pd
import numpy as np

# number of students
n = 200

# generate random study hours between 1 and 10
hours = np.random.uniform(1, 10, n)

# generate scores with a linear pattern + small noise
scores = 9.5 * hours + 12 + np.random.normal(0, 5, n)

# create dataframe
data = pd.DataFrame({
    "Hours": hours,
    "Score": scores
})

# save to csv
data.to_csv("student_scores.csv", index=False)

print("Dataset created: student_scores.csv")