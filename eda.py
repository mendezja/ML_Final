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
from sklearn.covariance import EllipticEnvelope
from sklearn.utils import resample
from sklearn.svm import SVC
from pandas import to_datetime
from scipy import stats


class EDA(object):

    def __init__(self):
        # Removed columns that had too many missing data or didn't logically make sense to include 
        self.columns = ['CONTRACTOR','STONE COLOR', 'DATE INSTALLED','PLACE INSTALLED', 'SQFT','PROJECT COST', 'MATERIAL COST', 'DEPOSIT', 'PAYMENT DATE']
        self.df = pd.read_csv("production19-21.csv", usecols=self.columns)
        print(self.df)
        
    def prepare_data(self):
        df = self.df

        # removing outliers for training data
        envelope = EllipticEnvelope(assume_centered=False, contamination=0.01, random_state=None,
                                    store_precision=True, support_fraction=None)
        pred = envelope.fit_predict(df)

        for i in range(len(pred)):
            if pred[i] == -1:
                # print(df.iloc[[i]])
                df.drop(index=i, inplace=True)

        # No Nan values means no imputation needed
        # print(df.isna().sum())

        # Drop all features except 1, 2, 4, 5 from feature selection
        df = df.drop(df.columns[[2,5,6,7]], axis=1)
        self.columns = self.df.columns

        self.df = df

    def feature_selection(self):
        '''Sequential Backward Selection'''
        X, X_test, y,  y_test = self.train_test_split()
        y = y.flatten()

        scaler = MaxAbsScaler().fit(X)
        X_scaled = scaler.transform(X)

        clf = SVC(kernel="linear")

        sfs = SequentialFeatureSelector(clf,
                                        k_features=1,
                                        forward=False,
                                        floating=False,
                                        scoring='recall',
                                        cv=5, n_jobs=-1)
        sfs = sfs.fit(X_scaled, y)
        print(sfs.k_feature_names_)

        # feature_counts = Counter(selectedFeatures)
        # df = pd.DataFrame.from_dict(feature_counts, orient='index')
        # print(df)

        fig1 = plot_sfs(sfs.get_metric_dict(),
                        kind='std_dev',
                        figsize=(6, 4))

        plt.ylim([0.8, 1])
        plt.title('Sequential Backward Selection')
        plt.grid()
        plt.show()

    def draw_plots(self):
        df = self.df
        # col1 = self.columns[0:6]
        # col2 = self.columns[6:12]

        # scatterplotmatrix(df[col1].values, names=col1, alpha=0.1)
        # plt.tight_layout()
        # plt.title("Scatter Plot Matrix")
        # plt.show()

        # scatterplotmatrix(df[col2].values, names=col2, alpha=0.1)
        # plt.tight_layout()
        # plt.title("Scatter Plot Matrix")
        # plt.show()

        scatterplotmatrix(df.values, names=self.columns, alpha=0.1)
        plt.tight_layout()
        plt.title("Scatter Plot Matrix")
        plt.show()

        cm = np.corrcoef(df.values.T)
        heatmap(cm, row_names=self.columns, column_names=self.columns)
        plt.title("Pearson’s R")
        plt.show()

    def upsample(self, X_train, y_train):
        df_X = pd.DataFrame(X_train)
        df_y = pd.DataFrame(y_train, columns=['target'])
        df = pd.concat([df_X, df_y], axis=1)

        # frequency of mode
        m = (df['target'] == 0).sum()

        # minority classes
        df_min = df[df['target'] == 1]

        # majority class
        df_maj = df[df['target'] == 0]

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

        y = self.df.loc[:, self.df.columns == 'target'].to_numpy()

        # Split into testing and training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)

        # Upsample data, do this after split to prevent data leakage
        X_train, y_train = self.upsample(X_train, y_train)

        return X_train, X_test, y_train, y_test


def main():
    eda = EDA()


if __name__ == "__main__":
    main()
