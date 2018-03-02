import pandas as pd
import numpy as np
import resample as rs

data = pd.read_csv("score.csv")
bootstrap = rs.boot(data[["mec", "vec"]], np.var)
bootstrap.plot(col=1)