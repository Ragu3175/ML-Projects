import pandas as pd
import numpy as np

n = 200

hours = np.random.uniform(1,10,n)
sleep = np.random.uniform(5,9,n)
practice = np.random.randint(0,5,n)

scores = 8*hours + 3*sleep + 5*practice + np.random.normal(0,5,n)

data = pd.DataFrame({
    "Hours":hours,
    "Sleep":sleep,
    "Practice":practice,
    "Score":scores
})

data.to_csv("student_scores.csv",index=False)