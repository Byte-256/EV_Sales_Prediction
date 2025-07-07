# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("jainaru/electric-car-sales-2010-2024")

# print("Path to dataset files:", path)
#
import numpy as np
import pandas as pd

def main():
    data = pd.read_csv('./EV_2010-2025.csv')
    print("Hello from ml!")
    print(data.info())




if __name__ == "__main__":
    main()
