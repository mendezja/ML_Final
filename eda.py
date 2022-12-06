from datetime import date
from lib2to3.pgen2.token import DEDENT
from re import L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap, category_scatter
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
# from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.utils import resample
from sklearn.svm import SVC
from pandas import to_datetime
from scipy import stats
 
class EDA(object):

    def __init__(self):
        # Removed columns that had too many missing data or didn't logically make sense to include
        self.columns = ['CONTRACTOR', 'STONE COLOR', 'DATE INSTALLED',
                        'PLACE INSTALLED', 'SQFT', 'PROJECT COST', 'DEPOSIT', 'PAYMENT DATE'] #look into including po number, diposit date? (), total paid (represents cost of job to client)
        self.df = pd.read_csv("Production2019-2021.csv", usecols=self.columns)
        self.target = 'DAYS_TO_PAYMENT'

    def prepare_data(self):
        df = self.df

        # Bin Contractor, Stone Color, and Place Installed because too many unique values
        # Replace values with count less than minCount to OTHER
        # TODO experiment with minCount
        minCount = 200
        df.loc[df.groupby('CONTRACTOR')["CONTRACTOR"].transform(
            'count').lt(minCount), 'CONTRACTOR'] = "OTHER"
        # print("\nCONTRACTOR feature summary")
        # print(self.df["CONTRACTOR"].describe())
        # print("\nCONTRACTOR feature value counts")
        # print(self.df["CONTRACTOR"].value_counts())

        df.loc[df.groupby('STONE COLOR')["STONE COLOR"].transform(
            'count').lt(minCount), 'STONE COLOR'] = "OTHER"
        # print("\nSTONE COLOR feature summary")
        # print(self.df["STONE COLOR"].describe())
        # print("\nSTONE COLOR feature value counts")
        # print(self.df["STONE COLOR"].value_counts())
        # TODO drop STONE COLOR?

        df.loc[df.groupby('PLACE INSTALLED')["PLACE INSTALLED"].transform(
            'count').lt(minCount), 'PLACE INSTALLED'] = "OTHER"
        # print("\nPLACE INSTALLED feature summary")
        # print(self.df["PLACE INSTALLED"].describe())
        # print("\nPLACE INSTALLED feature value counts")
        # print(self.df["PLACE INSTALLED"].value_counts())

        # One hot encoding
        df = pd.get_dummies(df, columns=['CONTRACTOR','STONE COLOR','PLACE INSTALLED'])

        # Engineer new feature DAYS_TO_PAYMENT
        # Turn string into date
        df['DATE INSTALLED'] = pd.to_datetime(
            df['DATE INSTALLED'], errors='coerce')
        df['PAYMENT DATE'] = pd.to_datetime(
            df['PAYMENT DATE'], errors='coerce')
        df['DAYS_TO_PAYMENT'] = (
            df['PAYMENT DATE'] - df['DATE INSTALLED']).dt.days

        # Drop rows without date
        df = df[pd.notnull(df['DATE INSTALLED'])]
        df = df[pd.notnull(df['PAYMENT DATE'])]

        # Bin DAYS TO PAYMENT
        daysToPaymentBins = [-1000, 0, 28, 1000]
        # Bins; 'Before Installation', '1 Week', '2 Weeks', '3 Weeks', '4 Weeks', '1+ Months'
        daysToPaymentLabels = [0, 1, 2]
        df['DAYS_TO_PAYMENT'] = pd.cut(
            df['DAYS_TO_PAYMENT'], bins=daysToPaymentBins, labels=daysToPaymentLabels)

        # Remove PAYMENT DATE and DATE INSTALLED
        df.drop(['DATE INSTALLED', 'PAYMENT DATE'], axis=1, inplace=True)

        # Turn deposit into categorical
        df["DEPOSIT"] = df["DEPOSIT"].fillna(0)
        df.loc[df['DEPOSIT'] > 0, 'DEPOSIT'] = 1
        
        
        # convert columns SQFT and PROJECT COST from object to int
        df["SQFT"] = pd.to_numeric(df["SQFT"], errors='coerce')
        df["PROJECT COST"] = pd.to_numeric(df["PROJECT COST"], errors='coerce')

        # SQFT: NA -> 0
        # Since some jobs are "removals/fixes", missing values represent 0 sqft 
        df["SQFT"] = df["SQFT"].fillna(0) 

        # removing outliers for training data
        # TODO Test without outlier removal
        df = df.dropna()
        envelope = EllipticEnvelope(assume_centered=False, contamination=0.1, random_state=None,
                                    store_precision=True, support_fraction=None)
        pred = envelope.fit_predict(df)
        # Remove outliers
        df = df[pred == 1]

        # print(df)
        self.df = df

    def feature_selection(self):
        '''Sequential Backward Selection'''
        X, X_test, y,  y_test = self.train_test_split()
        y = y.flatten()

        scaler = MaxAbsScaler().fit(X)
        X_scaled = scaler.transform(X)

        clf = SVC(C=10)
        # clf = DecisionTreeClassifier()
        # clf = RandomForestClassifier()

        sfs = SequentialFeatureSelector(clf,
                                        k_features=1,
                                        forward=False,
                                        floating=False,
                                        scoring='accuracy',
                                        cv=5, n_jobs=-1)
        sfs = sfs.fit(X_scaled, y)
        print(sfs.k_feature_names_)

        # feature_counts = Counter(selectedFeatures)
        # df = pd.DataFrame.from_dict(feature_counts, orient='index')
        # print(df)

        fig1 = plot_sfs(sfs.get_metric_dict(),
                        kind='std_dev',
                        figsize=(6, 4))

        plt.ylim([0, 1])
        plt.title('Sequential Backward Selection')
        plt.grid()
        plt.show()

    def draw_plots(self):
        df = self.df
        numCols = len(df.columns)
        # col1 = df.columns[0:(numCols//2)]
        # col2 = df.columns[(numCols//2 +1):numCols]
    

        # scatterplotmatrix(df[col1].values, names=col1,figsize = (8, 8), alpha=0.1)
        # plt.tight_layout()
        # plt.title("Scatter Plot Matrix")
        # plt.show()

        # scatterplotmatrix(df[col2].values, names=col2,figsize = (8, 8), alpha=0.1)
        # plt.tight_layout()
        # plt.title("Scatter Plot Matrix")
        # plt.show()

        scatterplotmatrix(df.values, names=df.columns, alpha=0.1)
        plt.tight_layout()
        plt.title("Scatter Plot Matrix")
        plt.show()

        cm = np.corrcoef(df.values.T)
        heatmap(cm, row_names=df.columns, column_names=df.columns)
        plt.title("Pearson’s R")
        plt.show()

    def upsample(self, X_train, y_train):
        df_X = pd.DataFrame(X_train)
        df_y = pd.DataFrame(y_train, columns=[self.target])
        df = pd.concat([df_X, df_y], axis=1)
        # print(df[self.target].value_counts())

        # frequency of mode
        m = (df[self.target] == 0).sum()

        # minority classes
        df_min = df[df[self.target] == 4]

        # majority class
        df_maj = df[df[self.target] == 0]

        # upsample the minority classes
        df_min_upsampled = resample(
            df_min, random_state=1, n_samples=m-len(df_min), replace=True)

        # concatenate the upsampled dataframe
        df_upsampled: pd.DataFrame = pd.concat(
            [df_min_upsampled, df_maj])

        X = df_upsampled.iloc[:, :-1].to_numpy()
        y = df_upsampled.iloc[:, -1].to_numpy()
        return X, y

    def train_test_split(self):
        # Get preprocessed data from eda
        self.prepare_data()
        features = self.columns

        X = self.df.iloc[:, :-1].to_numpy()

        # Scale data (do this in pipeline instead)
        # X = self.scale_data()

        y = self.df.loc[:, self.df.columns == 'DAYS_TO_PAYMENT'].to_numpy()

        # Split into testing and training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)

        # Upsample data, do this after split to prevent data leakage
        # TODO Test upsample, diffrence from max to min: 325-87 = 238
        # X_train, y_train = self.upsample(X_train, y_train)

        return X_train, X_test, y_train, y_test


def main():
    eda = EDA()
    # eda.prepare_data()
    # eda.draw_plots()
    # X_train, X_test, y_train, y_test = eda.train_test_split()

    eda.feature_selection()
    
    # print(X_train)
    # print(y_train)


if __name__ == "__main__":
    main()
