import pandas as pd
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("patricklford/global-ev-sales-2010-2024")
# print("Path to dataset files:", path)

def main():
    data = pd.read_csv("data/IEA_Global_EV_Data_2024.csv")
    print(data.head(2))
    print(data.info())




if __name__ == "__main__":
    main()
