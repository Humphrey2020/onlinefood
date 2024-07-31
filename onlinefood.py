{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "11bc6248-438b-40dd-a9e0-da7984dad0b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import seaborn as sns\n",
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "from sklearn .preprocessing import LabelEncoder,StandardScaler\n",
    "from sklearn.model_selection import StratifiedShuffleSplit,train_test_split, cross_val_score\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier \n",
    "from sklearn.ensemble import AdaBoostClassifier \n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import confusion_matrix,classification_report\n",
    "from xgboost import XGBClassifier\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4a342dc6-fd6e-4c5c-a38f-a4fba0287663",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(388, 13)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_excel(r'/Users/Humphrey/Desktop/onlinefoods.xlsx')\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "90bc0998-b162-4e9b-a194-e67c00e701d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age                           0\n",
       "Gender                        0\n",
       "Martial stattus               0\n",
       "Occupation                    0\n",
       "Monthly Income                0\n",
       "Educational Qualifications    0\n",
       "Family size                   0\n",
       "latitude                      0\n",
       "longitude                     0\n",
       "Pin code                      0\n",
       "Output                        0\n",
       "Feedback                      0\n",
       "Unnamed: 12                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8d6e8c74-bf53-45f8-b89d-17f7b8738f2f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the shape of the data set is (388, 13)\n"
     ]
    }
   ],
   "source": [
    "print(f\"the shape of the data set is {df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1565777f-5044-4a58-9a0b-acf97c2d22a0",
   "metadata": {},
   "source": [
    "the age is positive skewed "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4fb87446-e135-4f0d-9b04-796ac762d0f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Age', 'Gender', 'Martial stattus', 'Occupation', 'Monthly Income',\n",
       "       'Educational Qualifications', 'Family size', 'latitude', 'longitude',\n",
       "       'Pin code', 'Output', 'Feedback', 'Unnamed: 12'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8a165f7e-7836-4a9a-a80c-45fb0c66c491",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(columns=['Pin code','Unnamed: 12'],inplace= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b5aae827-2104-4ff6-b1e2-fa39923852e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Family size</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>24.628866</td>\n",
       "      <td>3.280928</td>\n",
       "      <td>12.972058</td>\n",
       "      <td>77.600160</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.975593</td>\n",
       "      <td>1.351025</td>\n",
       "      <td>0.044489</td>\n",
       "      <td>0.051354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.865200</td>\n",
       "      <td>77.484200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>23.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>12.936900</td>\n",
       "      <td>77.565275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>24.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>12.977000</td>\n",
       "      <td>77.592100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>26.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>12.997025</td>\n",
       "      <td>77.630900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>33.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>13.102000</td>\n",
       "      <td>77.758200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Age  Family size    latitude   longitude\n",
       "count  388.000000   388.000000  388.000000  388.000000\n",
       "mean    24.628866     3.280928   12.972058   77.600160\n",
       "std      2.975593     1.351025    0.044489    0.051354\n",
       "min     18.000000     1.000000   12.865200   77.484200\n",
       "25%     23.000000     2.000000   12.936900   77.565275\n",
       "50%     24.000000     3.000000   12.977000   77.592100\n",
       "75%     26.000000     4.000000   12.997025   77.630900\n",
       "max     33.000000     6.000000   13.102000   77.758200"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e34cab53-b200-43d4-9885-f4688f08f973",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Martial stattus</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Monthly Income</th>\n",
       "      <th>Educational Qualifications</th>\n",
       "      <th>Family size</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>Output</th>\n",
       "      <th>Feedback</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>Female</td>\n",
       "      <td>Single</td>\n",
       "      <td>Student</td>\n",
       "      <td>No Income</td>\n",
       "      <td>Post Graduate</td>\n",
       "      <td>4</td>\n",
       "      <td>12.9766</td>\n",
       "      <td>77.5993</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>24</td>\n",
       "      <td>Female</td>\n",
       "      <td>Single</td>\n",
       "      <td>Student</td>\n",
       "      <td>Below Rs.10000</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>3</td>\n",
       "      <td>12.9770</td>\n",
       "      <td>77.5773</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>22</td>\n",
       "      <td>Male</td>\n",
       "      <td>Single</td>\n",
       "      <td>Student</td>\n",
       "      <td>Below Rs.10000</td>\n",
       "      <td>Post Graduate</td>\n",
       "      <td>3</td>\n",
       "      <td>12.9551</td>\n",
       "      <td>77.6593</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>22</td>\n",
       "      <td>Female</td>\n",
       "      <td>Single</td>\n",
       "      <td>Student</td>\n",
       "      <td>No Income</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>6</td>\n",
       "      <td>12.9473</td>\n",
       "      <td>77.5616</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>22</td>\n",
       "      <td>Male</td>\n",
       "      <td>Single</td>\n",
       "      <td>Student</td>\n",
       "      <td>Below Rs.10000</td>\n",
       "      <td>Post Graduate</td>\n",
       "      <td>4</td>\n",
       "      <td>12.9850</td>\n",
       "      <td>77.5533</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender Martial stattus Occupation  Monthly Income  \\\n",
       "0   20  Female          Single    Student       No Income   \n",
       "1   24  Female          Single    Student  Below Rs.10000   \n",
       "2   22    Male          Single    Student  Below Rs.10000   \n",
       "3   22  Female          Single    Student       No Income   \n",
       "4   22    Male          Single    Student  Below Rs.10000   \n",
       "\n",
       "  Educational Qualifications  Family size  latitude  longitude Output  \\\n",
       "0              Post Graduate            4   12.9766    77.5993    Yes   \n",
       "1                   Graduate            3   12.9770    77.5773    Yes   \n",
       "2              Post Graduate            3   12.9551    77.6593    Yes   \n",
       "3                   Graduate            6   12.9473    77.5616    Yes   \n",
       "4              Post Graduate            4   12.9850    77.5533    Yes   \n",
       "\n",
       "    Feedback  \n",
       "0   Positive  \n",
       "1   Positive  \n",
       "2  Negative   \n",
       "3   Positive  \n",
       "4   Positive  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5b4bb395-fdba-4d37-b190-65fc74596854",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender\n",
       "Male      222\n",
       "Female    166\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].value_counts()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "57f56e4a-28db-4d84-b2f1-85602d9b9a7b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Martial stattus\n",
       "Single               268\n",
       "Married              108\n",
       "Prefer not to say     12\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Martial stattus'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "57eb372a-0f8f-4998-9026-8e32433174dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Feedback\n",
       "Positive     317\n",
       "Negative      71\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Feedback'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "cfce791c-029a-476b-a6e1-28eb3468c066",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8098768724480562"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Age.skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4486a991-475a-4a25-a234-9f0e28212e38",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the  values count of the feed back Feedback\n",
      "Positive     81.701031\n",
      "Negative     18.298969\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "count_feed_back=df.Feedback.value_counts()*100/len(df)\n",
    "print(f\"the  values count of the feed back {count_feed_back}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f835a11a-7029-4e22-9a58-1c1f90831b1b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "labels_feed_back=['postive feed back','negative_feed back']\n",
    "values=[317,71]\n",
    "plt.pie(x=values,labels=labels_feed_back,autopct='%1.1f%%')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "970a2bd3-57a2-4c03-936a-ba6aa290a2ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the output values of the products Output\n",
      "Yes    77.57732\n",
      "No     22.42268\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "out_put=df['Output'].value_counts()*100/len(df)\n",
    "print(f\"the output values of the products {out_put}\")  # Since we have high amount of positve chances \n",
    "# it clearly shows that  our product is satistactory"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fc2c7ee8-8909-405b-a467-14f3897f7980",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Monthly Income\n",
       " No Income          187\n",
       " 25001 to 50000      69\n",
       " More than 50000     62\n",
       " 10001 to 25000      45\n",
       " Below Rs.10000      25\n",
       " Name: count, dtype: int64]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "[df['Monthly Income'].value_counts()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "da49f2b3-4d77-423c-b03d-b506b47113ec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Age', 'Gender', 'Martial stattus', 'Occupation', 'Monthly Income',\n",
       "       'Educational Qualifications', 'Family size', 'latitude', 'longitude',\n",
       "       'Output', 'Feedback'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ebaa0d03-744f-4366-8737-e1712a1a2340",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 388 entries, 0 to 387\n",
      "Data columns (total 11 columns):\n",
      " #   Column                      Non-Null Count  Dtype  \n",
      "---  ------                      --------------  -----  \n",
      " 0   Age                         388 non-null    int64  \n",
      " 1   Gender                      388 non-null    object \n",
      " 2   Martial stattus             388 non-null    object \n",
      " 3   Occupation                  388 non-null    object \n",
      " 4   Monthly Income              388 non-null    object \n",
      " 5   Educational Qualifications  388 non-null    object \n",
      " 6   Family size                 388 non-null    int64  \n",
      " 7   latitude                    388 non-null    float64\n",
      " 8   longitude                   388 non-null    float64\n",
      " 9   Output                      388 non-null    object \n",
      " 10  Feedback                    388 non-null    object \n",
      "dtypes: float64(2), int64(2), object(7)\n",
      "memory usage: 33.5+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "558e8cac-7d46-42fb-9900-94fef5819d24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Martial stattus</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Monthly Income</th>\n",
       "      <th>Educational Qualifications</th>\n",
       "      <th>Family size</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>Output</th>\n",
       "      <th>Feedback</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>12.9766</td>\n",
       "      <td>77.5993</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>12.9770</td>\n",
       "      <td>77.5773</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>12.9551</td>\n",
       "      <td>77.6593</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>12.9473</td>\n",
       "      <td>77.5616</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>12.9850</td>\n",
       "      <td>77.5533</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender  Martial stattus  Occupation  Monthly Income  \\\n",
       "0   20       0                2           3               4   \n",
       "1   24       0                2           3               2   \n",
       "2   22       1                2           3               2   \n",
       "3   22       0                2           3               4   \n",
       "4   22       1                2           3               2   \n",
       "\n",
       "   Educational Qualifications  Family size  latitude  longitude  Output  \\\n",
       "0                           2            4   12.9766    77.5993       1   \n",
       "1                           0            3   12.9770    77.5773       1   \n",
       "2                           2            3   12.9551    77.6593       1   \n",
       "3                           0            6   12.9473    77.5616       1   \n",
       "4                           2            4   12.9850    77.5533       1   \n",
       "\n",
       "   Feedback  \n",
       "0         1  \n",
       "1         1  \n",
       "2         0  \n",
       "3         1  \n",
       "4         1  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "encode=LabelEncoder()\n",
    "df['Gender']=encode.fit_transform(df['Gender'])\n",
    "df['Martial stattus']=encode.fit_transform(df['Martial stattus'])\n",
    "df['Occupation']=encode.fit_transform(df['Occupation'])\n",
    "df['Monthly Income']=encode.fit_transform(df['Monthly Income'])\n",
    "df['Educational Qualifications']=encode.fit_transform(df['Educational Qualifications'])\n",
    "df['Output']=encode.fit_transform(df['Output'])\n",
    "df['Feedback']=encode.fit_transform(df['Feedback'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "60955edc-e11d-4bcf-aad2-131b1d34b83e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Martial stattus</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Monthly Income</th>\n",
       "      <th>Educational Qualifications</th>\n",
       "      <th>Family size</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>Output</th>\n",
       "      <th>Feedback</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>388.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>24.628866</td>\n",
       "      <td>0.572165</td>\n",
       "      <td>1.412371</td>\n",
       "      <td>1.902062</td>\n",
       "      <td>2.713918</td>\n",
       "      <td>1.069588</td>\n",
       "      <td>3.280928</td>\n",
       "      <td>12.972058</td>\n",
       "      <td>77.600160</td>\n",
       "      <td>0.775773</td>\n",
       "      <td>0.817010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.975593</td>\n",
       "      <td>0.495404</td>\n",
       "      <td>0.895035</td>\n",
       "      <td>1.329722</td>\n",
       "      <td>1.491790</td>\n",
       "      <td>1.035695</td>\n",
       "      <td>1.351025</td>\n",
       "      <td>0.044489</td>\n",
       "      <td>0.051354</td>\n",
       "      <td>0.417611</td>\n",
       "      <td>0.387157</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.865200</td>\n",
       "      <td>77.484200</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>23.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>12.936900</td>\n",
       "      <td>77.565275</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>24.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>12.977000</td>\n",
       "      <td>77.592100</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>26.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>12.997025</td>\n",
       "      <td>77.630900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>33.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>13.102000</td>\n",
       "      <td>77.758200</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Age      Gender  Martial stattus  Occupation  Monthly Income  \\\n",
       "count  388.000000  388.000000       388.000000  388.000000      388.000000   \n",
       "mean    24.628866    0.572165         1.412371    1.902062        2.713918   \n",
       "std      2.975593    0.495404         0.895035    1.329722        1.491790   \n",
       "min     18.000000    0.000000         0.000000    0.000000        0.000000   \n",
       "25%     23.000000    0.000000         0.000000    0.000000        1.000000   \n",
       "50%     24.000000    1.000000         2.000000    3.000000        3.000000   \n",
       "75%     26.000000    1.000000         2.000000    3.000000        4.000000   \n",
       "max     33.000000    1.000000         2.000000    3.000000        4.000000   \n",
       "\n",
       "       Educational Qualifications  Family size    latitude   longitude  \\\n",
       "count                  388.000000   388.000000  388.000000  388.000000   \n",
       "mean                     1.069588     3.280928   12.972058   77.600160   \n",
       "std                      1.035695     1.351025    0.044489    0.051354   \n",
       "min                      0.000000     1.000000   12.865200   77.484200   \n",
       "25%                      0.000000     2.000000   12.936900   77.565275   \n",
       "50%                      1.000000     3.000000   12.977000   77.592100   \n",
       "75%                      2.000000     4.000000   12.997025   77.630900   \n",
       "max                      4.000000     6.000000   13.102000   77.758200   \n",
       "\n",
       "           Output    Feedback  \n",
       "count  388.000000  388.000000  \n",
       "mean     0.775773    0.817010  \n",
       "std      0.417611    0.387157  \n",
       "min      0.000000    0.000000  \n",
       "25%      1.000000    1.000000  \n",
       "50%      1.000000    1.000000  \n",
       "75%      1.000000    1.000000  \n",
       "max      1.000000    1.000000  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "88a8a672-82a5-4a37-8908-ebbc5cbbe0da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender\n",
       "1    222\n",
       "0    166\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].value_counts()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "398c5d41-c5d1-4386-96db-4a7d8a17ff65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.2928475104718749"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Gender.skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8cca9090-f2f7-4f47-8809-ca74a2c974f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(df.Gender,shade=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b245757e-c8b4-4ab4-b9d3-e11b05c42e40",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Martial stattus\n",
       "2    268\n",
       "0    108\n",
       "1     12\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Martial stattus'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "f1e2bc56-e0f2-4c6b-8f45-d95beac66d58",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA6QAAAJZCAYAAAC+xPiuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzddXgUx//A8ffl4h7iwRJI8GCB4u4Udy/W4hSXIsUppUJx2kIpTmlpkeIS3BIIwUKUuLvr/f5IuOSSS4LkSL/85vU8+zy5vZm9+WT2dm92ZmclMplMhiAIgiAIgiAIgiB8YGrlXQBBEARBEARBEATh/yfRIBUEQRAEQRAEQRDKhWiQCoIgCIIgCIIgCOVCNEgFQRAEQRAEQRCEciEapIIgCIIgCIIgCEK5EA1SQRAEQRAEQRAEoVyIBqkgCIIgCIIgCIJQLkSDVBAEQRAEQRAEQSgXokEqCIIgCIIgCIIglAvRIBUEQRAEQRAEQRDKhWiQCoIgCIIgCIIgfGSuX79O7969sbGxQSKR8M8//5Sa59q1azg5OaGtrU21atXYuXOnysspGqSCIAiCIAiCIAgfmeTkZBo0aMDWrVvfKL2fnx89e/akTZs2PHr0iK+++oqZM2fy119/qbScEplMJlPpJwiCIAiCIAiCIAjlRiKR8Pfff9OvX79i0yxcuJCTJ0/y4sUL+brJkyfz+PFj7ty5o7KyiR5SQRAEQRAEQRCE/wHp6ekkJCQoLOnp6WWy7Tt37tC1a1eFdd26dcPFxYXMzMwy+Qxl1FW2ZUH4SGRG+ZZ3EVTqTL2l5V0ElRoYc628i6Byxtp65V0ElWprUqu8i6ByAzINy7sIKlVLLam8i6BSHjn65V0EldPNySnvIqjUKe2y+UH/X2Xw/+An/+ZXR8u7CEqV9e/I9Vv3sXLlSoV1X3/9NStWrHjvbYeFhWFpaamwztLSkqysLKKiorC2tn7vz1Dm4987BUEQBEEQBEEQPgKLFy9mzpw5Cuu0tLTKbPsSiUTh9eu7OwuvL0uiQSoIgiAIgiAIgqAKOdllujktLa0ybYAWZGVlRVhYmMK6iIgI1NXVMTU1VclngmiQCoIgCIIgCIIgqIbsf2e4e4sWLTh16pTCugsXLtCkSRM0NDRU9rliUiNBEARBEARBEISPTFJSEm5ubri5uQG5j3Vxc3MjICAAyB3+O2bMGHn6yZMn4+/vz5w5c3jx4gV79uxh9+7dzJs3T6XlFD2kgiAIgiAIgiAIqlCOE4K5uLjQoUMH+evX955+9tln7N27l9DQUHnjFMDOzo4zZ84we/Zstm3bho2NDZs3b2bgwIEqLadokAqCIAiCIAiCIKiArByH7LZv314+KZEye/fuLbKuXbt2PHz4UIWlKkoM2RUEQRAEQRAEQRDKheghFQRBEARBEARBUIWP/Bm+ZUE0SAVBEARBEARBEFThf2iW3fIihuwKgiAIgiAIgiAI5UL0kAqCIAiCIAiCIKhCTnZ5l+A/TzRIhf+s27dv06ZNG7p06cK5c+fKuzgq5eL2hN8O/clzD28io2P4af0yOrVtWd7FKsJ2bGfsp/ZC28KYxJfBPFm+j5h7L4tNb9qiFvVWjMagZkXSwuPw3naKV/suK6RRN9SlzuIhWPdsioaRHikBkTxdeZCIy24AOMzog/WnTTGwtyE7LYOYB148X3OYJJ9QVYaqYPmyOUycMBITEyPu33/EjC+X8Py5Z4l5jIwMWb1qIf379cDExAi/V4EsWLCKs+euyNPY2Fixft1XdO/WER0dbTy9fPnii7k8fPRE1SEVsWDxDMaMHYKRsREPXR6zYO5KXnp4F5t+2Ij+bN25ocj6iub1SE/PAEAqlbLgqxkMGtwbC0tzwsMiOXLoON9/u73EWf9UYeis4XQZ0RU9I328Hnnyy7KdBHoFFpu+WfcWDJw2COuq1kg11An1C+HkL/9w7W9neZo6n9Sl76T+VHesTgVLU775fC33L9wr03LX+Kwzdab0RMfCmDjPYFyWHyDyfvHfOYvmtXBaMRLjGhVJCY/j+fbTeO2/opCmcs+mNFgwCIOqFiT6R/D4m2MEnnORv6+up02DBYOo3KMJ2qaGxD57hcuyA0Q/9gVAoi6l4cJB2HRsiEFVczISUgm78ZRH646SGh5XpvEDmI/pgdXkfmhYmJDqGUjgit0k3X+uNK2GhQmVlo9Dz7E6WnbWROz5l8AVuxXS1Dy2BoMW9Yrkjbvsgvdna8q8/OVRhwA6ViY0XjIMmw71kepokuAbxt05vxDz5JU8jaG9DY2XDsOieS0kahLiXgZzY/IWUoKj3yvmamM745B3rkh4GYz78n1El3CuMGtRC8cVozHMO1d4bjuFX6FzxWuV+rbgk10zCDnrwt1xP8jXd3vwE3qVzYuk9/ntAo8X732veN5U31lDaDe8C3pGevi6ebF/2a+ElHCcaTusM60GtKNizSoAvHriy18bD+L3OP/Y23fWEPrNGqqQLz4ylllNJ6omiLfUY9YgWg7vhI6RPv5uXhxbtocwr6Bi07cY1pFPBrTFumZlAAKf+HFq42ECHvt8qCJ/eGLIbqlEg1T4z9qzZw8zZszg119/JSAggCpVqpR3kVQmNTWNmvbV6NezK7OXlP0PorJg07c5jqvG8HjRHmIeeGI7uhMtDi3kStv5pCr58aJbxZzmBxfgf+AqrtO3UaFpDRp8M5706ARC/30AgERDSss/FpMelcCDiT+RGhqDjo0pWUmp8u2YtqiN328XiXPzQSKVUnvxEFocXcSVtgvITklXedzz501l1pdfMH7ibLy8fPlq8ZecO3OYOvXakpSUrDSPhoYG584eJjIimqHDviAoOJTKlWxILJDe2NiI687/4HztNr16jyIiMorq1WyJi09QeUyFzZj1OVOmjWP6lEX4ePsxd8FU/jrxG82duhcbI0BCfCLNnboprHvdGAWYOftzxo4fzvTJC/F44UXDRvXYsn09CQmJ/Lxjn8riKaz/5AH0ntiXLfN+ItQ3mEEzhvD1wVVM7zCVtORUpXmS4hL5a+sxgnyCyMrIokmnpkz/7kvio+Nxu/4IAC1dLV698OPKscss3LW4zMtdtU8znFaO4sFXe4m474nD6I50PDifU+0XKm0w6FU2p+OBeXgddObW9B1YfFKDpuvGkhadSOCZ3O+cmZM9bXZO5/G3fxJ4zoXK3ZvQZtd0zvdbTfSj3B+Ezb+fiHHNStyesYOU8DiqDWxFp6OLONV+IalhsajraFLB0ZYnm/4h7nkAmka6OK0cTfu9czjbY3mZ/g9Merei8orxBCzZRdIDD8xHdcNh/zKedZhBRkhUkfQSTQ2youMJ3XwMy8/7KN2m9+ffINHI//mjbmJA3QubiD19u0zLDuVXh5pGunQ7sZzw2y+4MmojaVEJGNhakpGQIv8s/aoWdPtnGd5HrvH4u7/ITEjByKEi2WmZ7xVzxb7Nqb9qDG6L9hD9wBO70Z1odWghF0s4V7Q8uIBXB67iMn0bpk1r0DDvXBGSd654TaeSGY5fjyDqzosi27nafSkStfw70QxrVabNsa8IPlW2F4mK03NyP7pN6M3ueVsJ8wuh94xBzDuwnK86ziAtOU1pnlrN63L35E28H74kMz2TnpP6Mm//cpZ0mUVceIw8XdDLADaOWil/Lcv+bzRwOk/uQ4cJn3Jg3g4i/ULpOmMA0w4sYU3H2aQXE7ND87q4nryNX17MnSf1Yer+JazvMpf48NgPHIHwXyHuIRX+k5KTk/njjz+YMmUKvXr1KvKcpJMnT+Lg4ICOjg4dOnTg999/RyKREBcXJ09z+/Zt2rZti46ODpUrV2bmzJkkJxf/47o8tWnRlJlffEaX9q3KuyjFsp/UE//DzgQccibJK4Sny/eTGhyN7Wedlaa3HdOJ1KBoni7fT5JXCAGHnPE/7Iz9lF7yNFWHt0fTWJ/7Y38g5oEnqUFRxNx/ScLz/Ic03x2xgcCj10l8GUzC8wAezdqFbiVzjOvbqTxmgJkzJrL+m838889Znj17ybjxs9DV1WH4sP7F5hk3dhgVTIwZMGg8t++4EBAQzK3bD3B3z+/VWTB/KkFBIUz8fA4PXNzw9w/iytWb+Pr6f4iwFEye+hk/fLeDf09dwOOFF9MmLUBHR4eBg3uVmE8mkxEREaWwFNT0k0ac/fcSF887ExgQzKkT57l65RYNGzmqMpwiek3ow19b/+DeuTsEeAawee4mtLS1aNu3bbF5nt19yr3zdwn2DiI8IIx/fzuFv8crajetI0/zyPkhh787yL1zd1RS7tpf9MDnsDPeh5xJ8A7B9esDpIREU2NMJ6Xpa4zpSHJwNK5fHyDBOwTvQ874HLlGnck95Wlqfd6d0OtPebb1FAneoTzbeoqwm8+p/Xl3AKTaGlTp2ZRHa44Qce8lSa/Ccf/+OEmBkfLPzUxM5fKwDQScukeCTyhRD31wWboP0wbV0K1oWqb/A8sv+hJ15BJRhy+R5h1E4IrdZIREYT6mu9L0GUERBH69m+i/nMlOTFGaJjsuiazIOPli2KYhOanpxJ6+VaZlh/KpQ4A603qTEhLDndk/E+3mS3JQFGE3n5HkHyFP03DRYIKvPObRmiPEPvUnKSCS4MtupEe/30Uxh0k9eXXYmVeHnEn0CsF9+X5SgqOpVsy5wm5MJ1KConFfvp9ErxBeHXLm1WFnHKYUOv6oSWi6bRrPN/5FckBEke1kRCeSHhkvX6y7NCLJL4yo20Ubr6rQZXwvTm/7C9fz9wj2DOTXuVvQ0tGied82xeb5edZPXD1wnsDnrwjzCea3RTuRSCTUaaV4jMzJziYhMk6+JMZ8+AuXyrQb35ML2/7G/fx9Qj0DOTh3Gxo6Wjj1bV1snn2ztnDzwAWCn/sT4RPC4UW7UJNIqNHqw54XPqicnLJdPkKiQSr8Jx09epSaNWtSs2ZNRo0axW+//SYf4vfq1SsGDRpEv379cHNzY9KkSSxZskQh/5MnT+jWrRsDBgzA3d2do0ePcvPmTaZPn14e4fzPk2hIMapvR6Szu8L6iGtPqNC0htI8Jk4ORFxTHHoa6eyOcQM7JOpSAKy6ORHj4kX99ePo9mQHHZw34DCzL6hJii2LhoEuABlxSe8T0huxs6uCtbUlFy9dk6/LyMjg+o27tGjRpNh8vXt14e49V7ZsXktwoBtujy6zaOEM1Apcve/Vqyuuru4cObyLkKDHPLh/ngnjR6g0HmWq2lbG0soC5ys35esyMjK5fes+TZs1LjGvnr4uj55exf3FdQ79sQvH+rUV3r97x5W27VpQ3d4WgLr1atGshROXLjiXdRjFsqxsiYlFBdxuuMnXZWVk8ezeM2o61S4+YyGOrepjU60iz+89U0Epi1LTkFKhvh2h154qrA+99hTzJg5K85g5ORRN7/wE0wLfOXMne0ILfS9DnN0xy9umRCpFTV1KdrpiL1l2agYWn9QstrwahjrIcnLIjFfeCHwXEg119Byrk3DdTWF9wnU39JvUKrPPMRvemZiTN8lJLdsRF+VVhwCVujYm+rEvbXbNYJD7NnpeWIP9iPb5GSQSKnZqSKJvGB0PLWCQ+za6n15Bpe5O7xFx7rnCuL4dEW9xrjBVcq6IcHbHpEDMALXnDiAjOgH/w85vVI7KA1vjf/haqWnLgnllS4wtTHh647F8XVZGFi/vPcPeqfjvTWFaOppINaQkFzq/Wdpa88O9X/j2xnYmb5mNeWXLMiv7uzKtbIGRhQkeN/LrOisjC597z7FzUl7XymjqaKGmoU7KBzinlxeZLKdMl4+RaJAK/0m7d+9m1KhRAHTv3p2kpCQuX869n2Tnzp3UrFmTjRs3UrNmTYYNG8bYsWMV8m/cuJERI0Ywa9YsHBwcaNmyJZs3b2bfvn2kpSkfRiIUT6uCAWrqUtIi4xXWp0fGo21upDSPtoUx6YXSp0XGo6ahjmYFAwB0q1hg0+sTJFIJd0d+i+eP/2A/uSc1ZvUrtix1V44i+q4HiR7F36NSVqwsLQAID1fs+QsPj8TKsui9Sq/ZVavKwAGfIpVK6d1nNOvW/8TsWZP4avFMeZpqdlWYNGk03t5+9Ow1gp9/3s+mH1cxatQg1QRTDAsLMwAiIxSH0kVGRGNhaVZsPi8vX2ZMWcSoYVP4Yvwc0tLS+ffCEapVrypPs/nHnzn+57/ccTlHaPQzrt78h13bf+f4n/+qJhgljC1MAIiLjFNYHxcVh7G5cYl5dQ10Ofj8KH94H2fJnuX8+vXPPL7pppqCFvL6O5capfgdSo2MR8fCWGkeHXMjUgt951Kjcr9z2nnfOW1zY9IKbTMtKh6dvO9xVnIakS6eOM7qh46lMRI1CXYDWmHWuDo6lso/V01Lg0ZfDeXV33fITFI+BPpdqFcwQKIuJbNQ3WVGxqNhblImn6HX0AHdWlWJOnyxTLZXUHnVIYBBFXNqjOlEol84l0d8i9e+KzRZPQa7Qbk9V9pmhmjo61B3ei9CrrpzefgGAs+50u7XL7Fo/u6N/Xc5V2iVcK7Qyou5QtMa2A5vz8N5v75ROWx6NEHDSBf/ox+mQWqUdyxJKLSvxkfGY/QW++qghaOIDYvh2a38Rp6vmxe/zNnCD2NWs3fRTozMjVlyfC16xvplUfR3ZiiPWbHuEiLj5e+9iT4LRxAfFsPLWx9+7gThv0PcQyr857x8+ZL79+9z/PhxANTV1Rk6dCh79uyhc+fOvHz5kqZNmyrk+eSTTxReu7q64u3tzcGDB+XrZDIZOTk5+Pn5Ubu28p6R9PR00tMVr5KrpaejpaVVFqH97ys0D41EAiXOTVPoTYlEorBeoiYhPSoBt3m/Qo6MeHc/tK1MsJ/6KZ4//F1kc/XXj8WoThVu9FlZ5L2yMHx4f3Zsy5+op0/fMXnFLRpHSZPyqKmpERERzeQpC8jJyeHhoyfYWFsxd85k1qzdJE/j6urO0mXfAODm9ow6dWow+YsxHDjwZxlHlm/QkN58t2mV/PWIwV8Abx+j64PHuD7I7w24d9eVKzf+YeKk0Xy1IPc+6P4DP2Xw0D5MmjAXjxde1Ktfm7XffEVYWARHDxWt37LQtl87Jq2bKn+9dtzrWAvHV2RVEalJqcztMQttPW3qt2rAuKXjCQ8I49ndpyVnLEtF6qVoXZWWPnd1gfVFsit+kW/N2EmLHz5n4KOt5GRlE/PkFa/+voOJo22Rj5OoS2mzYxoSNTXuq2riGKXHnbKZFMtsWGdSPPxJdvMqk+0pVQ51iJoaMe6+uH3zBwCxT/0xqlmRGmM64ffnTSR5o1ACzz/E45fcSQNjnwVg3sSBGmM6EXHX443DUx5D0eKV9H0r8v/IC1omk6Gup03TbVN5OO9XMmIS3+jjbYd3IPzKY9JUMMkWQPO+bfhs3ST5603j1wHKjqNvvq/2mNSXZn1as2HY12QVGKHwxPlRfqKXAXg/fMm317fRamAHLuw+9R5RvJ0mfVszdN3n8te7xueeu5Se598w5k6T+tC4Tyu2DFupEPNH5yMdZluWRINU+M/ZvXs3WVlZVKxYUb5OJpOhoaFBbGwsMpksv2FT4P2CcnJymDRpEjNnzqSwkiZHWr9+PStXKjZ2ls6fyfIFX75LKB+N9JhEcrKy0bZQvMKtaWZEeqEr9a+lRcShVagXQMvMkJzMLDJik+RpZJnZkJNff4lewWhbmiDRkOa+l8dx7WdYdXXiZv9VpIXGoAqnTl3g/v38k7+WliYAVlbmhIXl37NkYWFGeETRCVVeCwsNJzMzi5wCJyEPDy+srS3R0NAgMzOT0NAInr9QnKnXw8ObAf17Ft5cmTp35gquLvkNSU3N3BgtLM0ID4+Urzczr1Ck17QkMpkMt4dPqFbdVr5uxeoF/PTjz/z9V26P6IvnnlSubMOsOZNU1iC9f/E+no/y/68amrmnOWNzE2Ij8ifMMDI1Ji4qrsRtyWQywvxzZ3N+9dyPSvaVGDB10AdpkL7+zukU6mnQNjMq0vv0mrKeN21TI3Iys0h//Z2LjCvSU6VtZkhqVP49aUn+EVwcuBapjhaaBjqkRsTReud0kgMiFfJJ1KW02TUD/crmXByyvkx7RwGyYhKRZWWjUSgmdTMjskqpuzehpq2JSZ/WhHx/+L23pUx51mFqRBzxniEKaeK9QqjSs2l+2TKziPcMLpQmGPMShmaXprhzhZaZUZFeXXmeiDi0C8dc4FxhWLMSelUsaLFvnvz91w3qfkH7udhqLskF7o3VqWSGRdt63B3/4zvHURq3Sw/wLXARQ11TAwAjCxPiC/SSGpoZkfAG+2r3z/vQa9pANo5cSZBHyfMIZKSmE+QRgKWd9TuV/V09ueTCKyUxG1oYK/QMG5gZklBMXRfU8fNedJnWj20j1xDiEVBq+v9pH+kw27IkhuwK/ylZWVns27eP77//Hjc3N/ny+PFjqlatysGDB6lVqxYPHijOvOfiojjdfePGjXn27Bn29vZFltc/wJVZvHgx8fHxCsvCLyerJNb/JbLMbOLd/TBvpzjpgEW7esQ8UP74k1hXLyzaKT5awbx9feIe+yHLym1oxtz3RM/OMr8LANCvZk1aWKxiY3TdWKx7NuXWoLWkFPpRXJaSkpLx8XklX54/9yQ0NJzOnfInv9HQ0KBtm+bcueNS7HZu33GhenVbhQsnDg7VCAkJIzMzMy/NA2rWqK6Qr4ZDNQICFH8glrWkpGT8fAPky0sPb8LDImjfIX9CLQ0NDVq2+oQH9x6+1bbr1a9NeIGGu46uNrJCV4azs3NQK+Ee4feVlpxKmH+ofAn0CiQ2IoYGrRvK06hrqFO3WV1eur7lZCcSCRp5P8JULSczmxh3P6zaKn6HrNrWI9JFeW9elKtXkfTW7eoRXeA7F+nqjXWRNI5EKdlmdmo6qRFxaBrpYtPOkcDzrvL3XjdGDe0suTT0G/lFprIky8wi+YkPhm0aKqw3bNOQJJf37MEDTHq3Rk1Tg+i/VDOsszzrMPKBJ4bVFRsshtWsSA6Okpct+rFvkTQG1axJDir+YltpZJnZxLn7YfEW54poJecKi/b1ic2LOdE7hEvtF3Cl82L5Enr+IZG3nnOl82JSQhQvnNkOa0d6VDxhlx6hKmnJaUT4h8mXEK9A4iJiqdu6vjyNVEOdms3q4u1a/ONuALp/0ZfeMwbx/WerefWk9EefqGuqY21fifgCF9g+hPTkNKL8w+VLmFcQ8RGx1FSIWUr1ZnXwcy35sWgdv+hNtxkD2fnZegKf+Kq66ML/ANEgFf5TTp8+TWxsLBMmTKBevXoKy6BBg9i9ezeTJk3Cw8ODhQsX4unpyR9//CGfhfd1A2DhwoXcuXOHadOm4ebmhpeXFydPnmTGjBklfr6WlhaGhoYKy4cYrpuSkoqHpw8enrkno+CQcDw8fQgNKzqTYHnx3nWGqiM6UGV4O/QdbKi3chQ6Fc3kzxWt/dVQGm+ZIk//at9ldCqZUXfFKPQdbKgyvB1Vh7fHe8dpeRq/3y+iaaKP45ox6FWzwrJzQxy+7Ivfbxfkaep/M47KA1vhOnUrWUmpaJkboWVuhJr2h2kYbN7yK4sWzqBv3+7UrVuTPbt/JCUllcNH8nv4ftvzE2vXLJK/3rlrH6amJvz4wyocHKrRs0cnFi2cwY6dv8vT/PTTLzRr1phFC2dQvbotw4b1Y+LEkWzfufeDxFXQzu2/M2vuZHr26kKt2g5s3fkNqamp/HUsv6627fqWpV/Plb+ev2g6HTq1pqptZeo51uanbeuo51iLvXuOyNOcP3uV2fOm0KVbeypXqUjPXl2YMn0c/54q+/v1SnJ690kGThtEs27NqVKjCtO//5L0tHSun7guTzPzh1mMXDBG/nrA1EE0aN0Qy8qWVKxekd4T+9J+QAeu/+MsT6Otq41tHTts6+TO+GxR2RLbOnaY2RR/7+3bePHzWexHtKf6sLYY2tvgtGIkehVN8cr7zjVcPISWP+UPG/TcdwX9SqY4fT0SQ3sbqg9rS/Xh7Xm+84w8jcev57Fu50idab0wtLemzrReWLepy4tf8p/1bN3OEev29dGrbI5V23p0/nMJCT6h+BzN/X9JpGq0/WUmpg3suDl9BxKpGtrmRmibG6GmkT8JTVkI//kEZsM7Yzq0E9r2laj89Xg0K5oRuf88ABUXjcJ2k+IIFp06dujUsUNNVxt1U0N06tih7VCpyLbNhnUm7vw9suPebBjouyivOvT4+RxmjatTd0Yf9G0tse3fAodRHXj52yV5mufbz1C1T3PsR7RH39aSGuO6UKlLIzx/z0/zLrx2ncF2RAeqDm+HgYMNjitHoVvRDN+8mOt+NRSnAucKv32X0a1khuOKURg42FB1eDtsh7fHK+9ckZOeSYJHkMKSmZBMVlIaCR5BChcvkUioOqwt/n/c+OCPRrm45zS9pg2kcbdPqFijMhO/m056ajp3T9yQp5n4/QwGLRgpf91jUl8GzB3OngXbiQqKxNDcGENzY7R0teVphn41hprN6mBWyYJqDR2Ytn0+Ovo63PrL+UOGp9S1PWfoMq0f9bs1xbpGZUZ+N5XM1HRcT+RPkjfq+2n0XjBc/rrTpD70mjuUQwt2EB0UgYG5EQbmRmjqfsS3RuVkl+3yERJDdoX/lN27d9O5c2eMjIpOfjBw4EDWrVtHbGwsf/75J3PnzuWnn36iRYsWLFmyhClTpsgbj/Xr1+fatWssWbKENm3aIJPJqF69OkOHDi2y3f+Cpx5ejJ+xUP762y0/A9C3R2fWLp1bXLYPKuTEXTRN9Kk5ZwBaFsYkegRxd+S3pOZdTde2NEanwCMfUgIiuTvyW+qtHI3duC6khcfyZOnv8meQAqSFxHB76DfUWzWKDle+IS0sFt9fzuG19aQ8jd3YLgC0/lvx+YYPv9xJ4NHrqNrG77ajo6PN1s3rMDEx4v79R/T4dITC8zmrVLZRGJ4bFBRCj54j+P67FTxyvUhwcBhbtu7m243b5GlcXB8zaPBE1qxZxNIls/B7FcicuV9z+LBqhrKWZMumX9DR0WbjD19jZGzEQ5fHDOo3XiHGSpWsFWI0MjLgh59WY2FpTkJCIk/cn9O7x0geueZPxrF4/moWLf2Sb7//GjNzU8LCIvj9tyN89802PqS/dx5HU1uLL9ZMRs9QHy83T1aN+lrhGaRmNubkFBg6rqWrxedrJmNqbUpGWgbBPkH8NOsHbp3O/6FVvb49q4+uk78evzz3QfVXjl1m67yf3rvc/ifvoWVigOPs/uhYGBP3MoirozaSnPcsRx0LY/Qq5jd+kwMjuTLqO5qsHEWNsZ1JDY/FZdk++fMrAaJcvLg5ZSsNFg6mwfxBJPmHc2PyVvnzKwE0DHVptHgIutYVyIhLJuDMfdy+OSbvodO1rkDlbrmzsfa6lB8/wMWBawlX8ozIdxV76hbqJobYzBqKhoUJqS8D8Bqzmozg3JESGhYV0KqoOMFY3Qv5QzX1Gthj2r8d6YERPGnxhXy9lp0NBs3q4Dn86zIrqzLlVYfRj325NmETDRcPpf7sfiQFRuKy/ACv/s5/1mrgORfuL9pD3el9aLJ6DAm+oVz//Cci75fcu1Wa4BN30TLRp9acAWhbGJPgEcStQucK3ULnitsjv6X+ytFUyztXPF76e5FnkL4Ji7b10K1k/kYz8Za1Mzv/QUNbk9Grv0DPSA8fNy++H71K4RmkphXNFG4x6ji6OxpaGkzfOV9hW/9sOsqJTbn3/5pYmzJp82wMTAxIjEnA55EXa/ovJjpYdaOF3tSlnSfR0NZk8OoJ6Brp4e/mzfbR6xSeQWpS0VRhZtjWo7ugrqXBhJ2Kv23ObjrG2U2qmz+hXIkhu6WSyMpqZgBBKEdr165l586dBAYGlvm2M6M+7uEkZ+otLe8iqNTAmA8zy2J5MtbWK+8iqFRbk7J7xMd/1YBMw/IugkrVUvt4H+kA4JFTvjOefgi6H/nELKe0y/axP/81Bv8P+qA2vzpa3kVQKv3F1TLdnlbtDmW6vf+Cj3/vFD5K27dvp2nTppiamnLr1i02btwonjEqCIIgCIIg/Ld85BdzyoJokAr/k7y8vFizZg0xMTFUqVKFuXPnsnjx4vIuliAIgiAIgiDkE0N2SyUapML/pB9//JEff1TdlO6CIAiCIAiCIKieaJAKgiAIgiAIgiCoghiyWyrRIBUEQRAEQRAEQVABmezjfFRLWRLPIRUEQRAEQRAEQRDKheghFQRBEARBEARBUAUxqVGpRINUEARBEARBEARBFcQ9pKUSQ3YFQRAEQRAEQRCEciF6SAVBEARBEARBEFRBDNktlWiQCoIgCIIgCIIgqEKOmGW3NGLIriAIgiAIgiAIglAuRA+pIJTiTL2l5V0Eler5dE15F0GlDjouK+8iqFyqmqS8i6BSalnlXQLVC/vIz8Zh6Jd3EVRK++P+CgIQrf5x92H0TNcu7yKoVKT6/4Od9L9KDNkt1Ud+ChQEQRAEQRAEQSgnYpbdUn3cl7sEQRAEQRAEQRCE/yzRQyoIgiAIgiAIgqAKYshuqUSDVBAEQRAEQRAEQRXEkN1SiSG7giAIgiAIgiAIQrkQPaSCIAiCIAiCIAiqIHpISyV6SAVBEARBEARBEFRAJssu0+Vtbd++HTs7O7S1tXFycuLGjRslpj948CANGjRAV1cXa2trxo0bR3R09LuG/0ZEg1QQBEEQBEEQBOEjc/ToUWbNmsWSJUt49OgRbdq0oUePHgQEBChNf/PmTcaMGcOECRN49uwZx44d48GDB0ycOFGl5RQNUkEQBEEQBEEQBFXIySnb5S388MMPTJgwgYkTJ1K7dm02bdpE5cqV2bFjh9L0d+/exdbWlpkzZ2JnZ0fr1q2ZNGkSLi4uZfGfKJZokAqCIAiCIAiCIKiCLKdslzeUkZGBq6srXbt2VVjftWtXbt++rTRPy5YtCQoK4syZM8hkMsLDw/nzzz/59NNP3+tfUBrRIBX+J7Vv355Zs2aVdzEEQRAEQRAE4YNJT08nISFBYUlPTy+SLioqiuzsbCwtLRXWW1paEhYWpnTbLVu25ODBgwwdOhRNTU2srKwwNjZmy5YtKonlNTHLrvDOwsLCWL9+Pf/++y9BQUEYGRnh4ODAqFGjGDNmDLq6uuVdxA/Odmxn7Kf2QtvCmMSXwTxZvo+Yey+LTW/aohb1VozGoGZF0sLj8N52ilf7LiukUTfUpc7iIVj3bIqGkR4pAZE8XXmQiMtuADjM6IP1p00xsLchOy2DmAdePF9zmCSfUFWG+lZc3J7w26E/ee7hTWR0DD+tX0anti3Lu1hK2X/WmZpTP0XHwph4z2AeLd9PVAl1aN6iFg1XjMKoRkVSw+Pw2H4anwJ1WLFnE+rM7Iu+rSVqGlISfcN5uesM/n/elKepPaMPlXo2kddhlIsX7muOkFhGddhwzgBqjOyAppEeUY98uLtkL3GewSXmqdqzKY3mD8KgqgWJ/hE83HCMgHOKQ3ZqftaZepN7omthTKxnMPe/PkDEfeX/qxYbxlNzVEfuf72f57+el683qGpBk2UjsPykBmqaGgQ7u3Nv6e+kRSW8UWw1P+tMnbwyxHkG86CEMgBYNq9Fk69HYlyjIinhcTzbcRrP/VcU0lTp2ZSGBWJ/tOEYgQVibzBnAA3mDlDIkxoRx7FG0xXS2PZtjq5NBXIysol54sejDceIeuTzRnGVpuXsAdQf0QEtIz3CHvlwadleokupU4ceTWk9bxBGVSyID4jgxrfH8D6fH5dEqkbL2QOo068luhbGJEfE8ezYde5sPgEymfxza/ZujqFNBbIzswl/4seNb48R5lY2cZVnfGrqUlrPH4Rdh4YYVzEnPTEV/5tPuf7NUZLD48o0PoBPZg+g7sjcGMMf+XBt6V5iSomxeo+mNJs3CKOqFsT7R3B34zF8zykfSuc0rTctFg3F7ddz3Fx5QOE9E3sbWn41DJtmtZCoSYjxDObclC0khZTdxCXNZg+g3ogOaOfV4dVlpcdn36MpzQvU4Z1vj+FToA6bzR5A89mK373kiDh+bZL/3dM1M6TV4mFUaeuIlqEuwfdecm3578S9Ci+z2KrnnSe0LYxJ8AzGrZTzhFneecIw7zzxcvtpfAudJ2oVOE8k5Z0nAgqcJ+rMHUDdeQMVtpsWEcepBtPKLK7Cms4eQJ0C++j1pXuJLaUOq/VoyicF9tF7G4/hV8w+2nhab5ovGsrjX89xq8A+2vGHL6g1uK1C2rCH3hzvu+K9YyoXZTzL7vr161m5cqXCuq+//poVK1YoTS+RSBRey2SyIutee/78OTNnzmT58uV069aN0NBQ5s+fz+TJk9m9e3eZlF8Z0SAV3omvry+tWrXC2NiYdevW4ejoSFZWFp6enuzZswcbGxv69OlT3sUsVnZ2NhKJBDW1shskYNO3OY6rxvB40R5iHnhiO7oTLQ4t5Erb+aQGFz3J61Yxp/nBBfgfuIrr9G1UaFqDBt+MJz06gdB/HwAg0ZDS8o/FpEcl8GDiT6SGxqBjY0pWUqp8O6YtauP320Xi3HyQSKXUXjyEFkcXcaXtArJTil4xKw+pqWnUtK9Gv55dmb1kTXkXp1iV+zSn4arRPFz8G5EPPLEf3ZG2Bxdwrt0CUpTUoV5lc9oemI/vwavcnb4d86Y1aLx+HOnRCQTl1WFGbDLPfzpBgncIORlZ2HRpxCc/fkF6VDxhzk+A3Eat12+XiHHzQU1diuOiIbQ7soizbReQnfp+dVhvai/qfNGDm7N3keAbRoMv+9L18CKOt51PVnKa0jzmTva02zGdRxv/JOCsC1V6NKH9zumc6b9a3qCy7dOMT1aM4u5Xe4l44EnN0R3pcmA+/7RfSHKhH7VVujlh3qg6yaExCuvVdbTocmghsc8DODdkHQCN5w+i0965/Nt7hbwRVBzbPs1osmIU977aS+QDTxxGd6TTgfmcVFIGAP3K5nTcPw+vQ87cnLED86Y1aLZuLGnRiQScya0vMyd72u6YjluB2NvtnM65ArEDxHoEcnHYN/LXsmzFHxwJvqHcX/o7if4RSLU1qfN5DzofWsjfreaSHpNYYlyl+WRKL5wm9uDc3F3E+obRfGZfBh9cxO7288kspk6tG9vTe9t0bn7/J97nXLDv3oTe26dzeOBqeWPykym9aDCqE+fm7CLKMwir+nZ0/+4L0hNTebgn9yJCjG8ol5f/TnxABOramjhN6MHgAwv5te1cUt8zrvKOT11HE4t6ttzd/A8RzwPQNtKlw9ej6b97Dgd6LS+T2F5rPKUXDT/vwaU5u4jzC6PJzL70PbSIA+2Kj9GqsT3dtk/n3nd/4nPOherdm9Bt+3SOD1hNeKELAhYNqlF3RAeinvsX2Y5hVQsGHl/G8yPXuPf9X2QkpmBiX5Hs9Mwyi89pSi8aTezBxbm7iPMNo+nMvvQ/uIh9JdShVWN7emybzp3v8+PrsX06xwYqxhf1MpC/RxT/3ev1y2xysrI5PeFH0pNSafx5D/ofWsz+TgvJes/jKUClAueJqAeeVBvdkTZ55wml5/rK5rTJO0/cm74dswLnieAC54kXP50gMe88Yd2lEU3zzhPheecJgHiPQK4NWZ8fuwofJ9JoSi8afN6DK3n7qNPMvvQ5tIhDJeyjlo3t6bp9Ove/+xPfcy5U696Ertun8/eA1UQo2UfrFLOPAvhffcyVuT/LX+dkZpVdcB/aWwyzfROLFy9mzpw5Cuu0tLSKpDMzM0MqlRbpDY2IiCjSa/ra+vXradWqFfPnzwegfv366Onp0aZNG9asWYO1tXUZRaFIDNkV3snUqVNRV1fHxcWFIUOGULt2bRwdHRk4cCD//vsvvXv3BiA+Pp4vvvgCCwsLDA0N6dixI48fP5ZvZ8WKFTRs2JD9+/dja2uLkZERw4YNIzEx/4dNcnIyY8aMQV9fH2tra77//vsi5cnIyGDBggVUrFgRPT09mjVrhrOzs/z9vXv3YmxszOnTp6lTpw5aWlr4+ys/CL4r+0k98T/sTMAhZ5K8Qni6fD+pwdHYftZZaXrbMZ1IDYrm6fL9JHmFEHDIGf/DzthP6SVPU3V4ezSN9bk/9gdiHniSGhRFzP2XJDzPnx3t7ogNBB69TuLLYBKeB/Bo1i50K5ljXN+uTON7H21aNGXmF5/RpX2r8i5KiWpO6oHfYWd8DzmT6BXCo+UHSA2JpnoxdVh9TCdSgqN5tPwAiV4h+B5yxu/INWpOzr/XIvLOC4LPupDoFUKyfwRev54n/kUAZp/UlKe5PuJbXv1xnQTPYOKeB3B/9i70KplRocH712Gdid1x33yCgLMuxL0M4sasXajraFKtf/E91HUmdifk+lOebD1FvE8oT7aeIvTmc+pM7C5PU/fzHngdccbrsDPx3iHc//oAySHR1BzTSWFbulYmNFv7Gdenb0eWpThdvUVTB/Qrm3Nz9s/EeQQR5xHEzTk/Y96oOtat65QaW+3Pe+B9xBnvvDK45JWhRqEyvFZjdEeSg6Nx+foA8d4heB92xvvoNepO7qkQe+j1pzzdeooEn1Ce5sVeu0DskPsjOC0yXr4UbmT6/XOH0BvPSAqIJN4zGJeVB9E01MWkTpVS4ypN4wndubf1BF7nXIjyDOLsnF2oa2tSu1/xdeo0oTv+N55yf9spYnxCub/tFAG3nuM0IT8uGycHfC644nvFjYSgKDzPPODV9SdYFjiWeJy4Q8DNZ8QHRBLtGYzz6oNoGepiXvv94yrv+DISU/lz5AZenr5HrG8ooY98uLJ8H1b1q2FgY1pm8QE0mNAdly0n8D3nQszLIC7Nzo2xRgkxNpjYncAbT3Hddoo4n1Bct50i6NZzGhTaNzV0tei6eQpXFu4mPT6lyHaaLxjMqyuPub3uCFHP/EkIiMT/ihup0W82KuFNNJrQnQdbT+BzzoVozyAuztmFhrYmNUuIr9GE7gTceIrLtlPE+oTisu0Ugbee02hCoe9eVg4pkfHypeCFEGM7K6ydHLiy5DfC3X2J8w3l6pLf0NDTombfFmUSW42884Rf3nni8fIDpLzBeeJx3nnCr5jzREiB84S3kvPE69jTI+PlS0Z02VwEUqb+hO64FthHL+ftow5vsI8+zNtHH247RbCSfVRdV4vOm6fgXMw+CpCdkUlqZLx8SY9LLtP4/pdpaWlhaGiosChrkGpqauLk5MTFixcV1l+8eJGWLZXXY0pKSpHOGqlUCuT2rKqKaJAKby06OpoLFy4wbdo09PT0lKaRSCTIZDI+/fRTwsLCOHPmDK6urjRu3JhOnToRE5PfU+Lj48M///zD6dOnOX36NNeuXeObb/Kvfs6fP5+rV6/y999/c+HCBZydnXF1dVX4vHHjxnHr1i2OHDmCu7s7gwcPpnv37nh5ecnTpKSksH79en799VeePXuGhYVFmf1PJBpSjOrbEensrrA+4toTKjStoTSPiZMDEdeeKKyLdHbHuIEdEvXcL79VNydiXLyov34c3Z7soIPzBhxm9gU15UMtADQMcodKZ8QlvU9I/++oaUgxqW9HWKE6Cbv2BLMmDkrzmDZxKJre2Z0KBeqwMIvWdTGobk3kXY9iyyKvw9j3q0P9KuboWhoTUqCMORlZhN31wKKYmCC3hzTkumJcwdfc5XnUNKSY1rcj5NpThTQh154qblcioc3myTzd8a/SIcJqWhogk5Gdkd8zk52eSU52DpZNaxZJr5C3mDKEXnuKeTGxmTs5EFq4zM5PMK2fX1/KYg+55l5kmwZ2lgxy3UL/Oz/QZvs09KuYl1hWh5EdyIhPJvbZ+10IM6pijr6FMa8KlDE7I4ugex5UdCq+Tm0a2yvkAXh1zV0hT/ADT6q0qouJnRUA5rWrULFpTfyuPEYZNQ0p9Ud0IC0+mchiejne1n8pPgBNQx1kOTmkJyj/0fwuDKuYo2dpTMB1xe9l8D0PrEuI0aqxvUIegIBr7lgVytNuzVheXXEj6OazohuRSLDt2JA4vzD6HFjA+EfbGHRyBXbdnN4rpoIMq5ijZ6EY3+s6LCk+62LiK5zH2M6SCQ+2MPbmD3TfOg3DAt89qWbuwL+Cvb2yHBk5mdnYFHMufhuSYs4T4aWcJ8KVnCdM3uE8oV/Nkl6PttLz3o802zEdvRKOO+/j9T4aWGgfDbnnUWR/K8iysb1CHlC+j7ZdMxb/4vbRPBWb12bso22MuLaR9hsmoGNq+I7R/AeU4yy7c+bM4ddff2XPnj28ePGC2bNnExAQwOTJk4Hc3tYxY8bI0/fu3Zvjx4+zY8cOfH19uXXrFjNnzuSTTz7BxsamTP8tBYkhu8Jb8/b2RiaTUbOm4g9GMzMz0tJyh3FMmzaNbt268eTJEyIiIuRXbr777jv++ecf/vzzT7744gsAcnJy2Lt3LwYGBgCMHj2ay5cvs3btWpKSkti9ezf79u2jS5cuAPz+++9UqlRJ/rk+Pj4cPnyYoKAg+Zdl3rx5nDt3jt9++41163KHAmZmZrJ9+3YaNGhQ5v8TrQoGqKlLSYuMV1ifHhmPtrmR0jzaFsZERCo2YNMi41HTUEezggHpEXHoVrHArFUdgo7f4u7Ib9G3s6L++rFI1NXw/OFvpdutu3IU0Xc9SPQIKpvg/p/QLKYO00qoQx1zI8KUpFfTUEerggFpEXEAaBjo0PvRVqSa6siyc3BdvJfw60+VbDFXwxUjibznQfzL96tDHQtjAFKjFMuYGhmPfiWz4vOZG5MaWTSPTt7/4fX+XmS7UfHyzwRwnNaLnKwcXuw+jzKRrt5kpaTTZMkwXNf/gUQiwWnJUNSkauhYGivN85r8O6ekDDYFyqAQl4VRkTKnReXWl3YFA1Ij4tA2N1a6D+gU2AciH3lz68tdJPiGomNuhOPMfvQ48TUnOy4ivcBFhIqdG9J2+3TUdTRJDY/j4vANCu+/Cz3z3NiSC8WRHBWPYcXi61TP3FhpHt0Ccd3ffgotAx3GX/2WnOwc1KRq3Nh4DI+TdxTyVevUkF5bp6Oho0lSRBx/jtxA6nvGVbCcr8tWuKwfKr7XpFoatF00lBf/3CGjwG0S70s3L0Zl30uDEr6XuubGSr9zegVidOjTHHNHW/4oZoixrpkhmvo6OE3txd2Nf3J73RGqtG9Az5+/5O+h6wgp4ULZm3pdhymFyppSSh3qmhsrzVOwDsMeeXNh9i5ifUPRNTfikxn9GHL8aw50XkRaXBKxPqEkBEbScuFQrizeTWZKOo0/74mehTF6xRwX3sbr4076W5wntM2NlP42KHyeUM87T6jlnSceLt5LRIHzRMwjH+7P3EmiTxja5obUntWPjqdWcL79wve+eFmYbnF1+I77aME6tM/bR/8sYRh8wNXH+Jy+T2JwFAaVzWk2bxB9ji7mWM9l5GT8Dw7dLeMhu29j6NChREdHs2rVKkJDQ6lXrx5nzpyhatWqAISGhio8k3Ts2LEkJiaydetW5s6di7GxMR07dmTDhg0qLadokArvrPAN0ffv3ycnJ4eRI0eSnp6Oq6srSUlJmJoqDnVKTU3Fxyf/XgJbW1t5YxTA2tqaiIgIILexmZGRQYsW+UNtKlSooNAYfvjwITKZjBo1FK9+pqenK3y2pqYm9evXLzGm9PT0IjOVZcqy0ZAov4pZRKHRDBJJKbfBFXpT/j/NWy9Rk5AelYDbvF8hR0a8ux/aVibYT/1UaYO0/vqxGNWpwo0+K4u8J7yhwhVWzI3/+ekLJ1esQ4DMpDQudP4KdT1tLFvXpeGKkST5RxB550WRzTVeNxbjOlW43HfVWxe9Wv+WtNgwXv760pjvii9jaSNvlOYptLLI/pu/ztTRljoTunGy+9JiPyI9JhHnSZtpvn4ctcd3RZYjw+/EHaLc/YrcF1ZsMUsoQzEZSt1OkSSFYg+5mn8hKc4jiEgXb/rf/p5qg9vw4uez8vfCb73gdNclaFXQx2FEB9runM7ZXitIe4uhkbX7taTL+vw6PT5WdXVas3dzavdvxekZ24n2DMKiblU6fD2K5PA4nv15Q54u8PYL9nVfgk4FfeoP70Dv7dM52HcFKe8w5PO/GB+AmrqUXlunIZGocWnp3reMSlGNfi1p/01+jKfzYlS2n5U2Iq7o+xL5vqtvXYE2K0ZzYuSGYu8HleSNrvG78JDHv54DIOp5ANZNHKg3qtM7NUhr9mtJxwJ1eLKY+CTvEl+hOvQvMAop+mUQoa7ejL3xPbUHteHRr2fJycrm38k/0fnbz5n85GdysrIJuPmMV1fc3jqukstZ9DxRYmjK4iq0naxC54kGK0aSXOA8EVagJz/BA6JdvOl59weqDmmD166zvA+HQvvov+/xPSxtH229YjSnSthHAbxP3ZP/HfMyiEh3P0bf2YRtx4bFTuIlFG/q1KlMnTpV6Xt79+4tsm7GjBnMmDFDxaVSJBqkwluzt7dHIpHg4aF44qpWrRoAOjo6QG7Pp7W1tcK9nK8ZGxvL/9bQ0FB4TyKRkJM3JOFNxqvn5OQglUpxdXWVj3N/TV9fX/63jo5OsbOKvaZs5rKhevUYru9YYr70mERysrLRtlC8QqppZkR6oauFr6VFxKFV6IqtlpkhOZlZ8qudaRFxyDKzISf//5DoFYy2pQkSDWnue3kc136GVVcnbvZfRVqhyWOE0mXI69BYYb22mWGRq9uvpUbGF6nz13Wo0BMmk5GUN8Nj3DN/DB1sqD2zT5EGaeM1Y6jYtTFX+q8m9R3qMODCQyILTLzzeviajrkRqXlX4V/HVPgqtmJccegUiis3T26D4/X+rpN3FV2extRI3rNq2awm2maGDL7/k/x9NXUpTZaPpM7E7vzZfDYAIdefcrzVXLRM9JFl55CRkMLQR1vxC4gsMdY3KUORuCLii6Y3M1Kor7RSYlcmKzWdWI9ADO0si6xPfBVO4qtwoh760O/md9gPb8fTradKjK0g74sPCS1Yp1q5dapnbkRygTrVNTUs0ptRUHJknEJPWn6e/LjaLRnO/e2neHnqLgBRL4MwrGjGJ1N7KzTYMlPTifMPJ84/nNBHPky49h31hrXj/rY3j+u/HJ+aupTe22dgVNmcP4atf+/eUb+LDxUm5Xn9vdQ1NyKlQIw6ZobF7rsAKZFxCj1Nr/O8jtG8vh265kYMPbNaIRabZjWpP7YLO6qPJTUmkezMLGK8FIfQx3gFY1PKMPni+F58SFgxdagQXyl1mPIGdVhYVmo60S8DMS7w3Yt48opDPZagaaCDVEOd1JhEhp5YQbi739uGVkR6CeeJwr2mr6WVcJ7IKHSeSM47T8Q/88fAwYZaSs4Tr2WnphP/IhCDvCHo7+PVxYccfcN9NOUd9tHUQvvoYCX7qOPYLuyqPhZZTtHffSkRcSQGR2FUBrGWCxVOPvWxEPeQCm/N1NSULl26sHXrVpKTi7/JvHHjxoSFhaGuro69vb3CYmZW/JCPguzt7dHQ0ODu3bvydbGxsXh6espfN2rUiOzsbCIiIop8jpXV2x28Fi9eTHx8vMIySK/0yVVkmdnEu/th3k6x4WrRrh4xDzyV5ol19cKiXT2Fdebt6xP32E8++UvMfU/07CwVeun0q1mTFhar2BhdNxbrnk25NWgtKaX8kBeUy8nMJtbdD6u2inVi2daRKBcvpXmiXbywbKtY51btHIkpUIdKSSTyE/5rjdd+RsWeTbk6eC3Jge9Wh1nJafIGUOKrcOI8g0kJj8OmQExqGlKsmtciopiYIHcorU0bxf+DTVtHeZ6czGyi3f0Utpubpp48jc9ftzjR+StOdl0iX5JDY3i2418ujPy2yGemxyaRkZCCVas6aJsZEnjxYYmxFlcG67b1iCwmtkhXL6wLl7ldPaLd8+sr0tUbayWxF7dNADVNdYwcch/nUDIJUk2NUtIoykxOkzf+4vzDifYMJikijqptFOu0UrNaBLsWX8aQh94KeQBs2zoq5NHQ0SzyYzAnJ0feq1YsiQT1t4zrtf9afK8boyZ2lhwb8Q1pZXAvfmZyGvGvwuVLjGcwyeFxVC4UY8VmtQgtIcawh94KeQCqtHUkLC9P0M1nHOq8iCPdl8iX8Me+vPz7Nke6L5HfSxnx2BfjaoozZRpXsyYxOOrd4/MPly8xnsEkR8RRRUkdlhRf6ENvhTyv4yspj1RTHRP7igoXL17LSEwlNSYRY1tLLOpXw/eCa9ENvCVZ3nnCsgzOE7GlnCckEglqmsX3G6lpqmPwRsed0mUmp5HwKly+xObto5UK1aFNs1ry/U2ZcCX7aOVC++iRzov4o/sS+RLx2BfPv2/zR94+qoyWsT761hUUGsf/U8rxHtL/FaJBKryT7du3k5WVRZMmTTh69CgvXrzg5cuXHDhwAA8PD6RSKZ07d6ZFixb069eP8+fP8+rVK27fvs3SpUtxcXmzIRf6+vpMmDCB+fPnc/nyZZ4+fcrYsWMVZgCrUaMGI0eOZMyYMRw/fhw/Pz8ePHjAhg0bOHPmzFvFpWzmsjcdruu96wxVR3SgyvB26DvYUG/lKHQqmsmfK1r7q6E03jJFnv7VvsvoVDKj7opR6DvYUGV4O6oOb4/3jtPyNH6/X0TTRB/HNWPQq2aFZeeGOHzZF7/fLsjT1P9mHJUHtsJ16layklLRMjdCy9wINe13+4GoCikpqXh4+uDhmXsFNjgkHA9PH0LDIsq5ZIpe7jqL3YgO2A1rh4GDDQ1XjkK3oqn8uaKOXw2l2ebJ8vQ++y6jV8mUhitGYuBgg92wdtgNb8/Lnf/K09Se0QfLtvXQq2KOgb01NSb1wHZwa/z/uiVP47R+LFUHtuLutG1kJaWhbW6EtrkR0jKow+e/nqP+jD5U6d4E45qVaP3jJLJSM/D9+7Y8TeufJtF40ZD8PLvPY9POkXpTe2FU3Zp6U3th06Yuz/OG+AE8++UsDsPbYz+0LUb2NjRdMRK9iqa83J/7v0qPTSLuZZDCIsvKJjUyjoQCz1e1H9IW88bVMahqQbUBrWi/awbPfjmnkKY4L345i32BMjTJK4NnXhkaLRpCq58mydN77r+CXiVTmnw9EiN7G+yHtsV+WHue7cw/TrzIi73u1F4YVrem7tReWLepy4sCsTstG45l81roVzbHrFF12v08Ew19HXyO5fayqeto0WjREMwaV0evoikV6tnSYuNE9KxNeHU6fyjau3q4+xzNpvXBvlsTzGpUosf3k8hKy+DFP/l12uPHSbRZmF+nD/ecx7atI59M6UWF6tZ8MqUXVVrXxXV3flw+lx7RfEZfqnVsiGElM+y7NaHJxB7yZ3lq6GjResEQrBtVx7CiKRb1bOm6YSIGVia8/Pf94yrv+CRSNfrsnIllfTv+nbkDiVQNXXMjdM2NUNN4w9s23tDj3edoMr0P1bo3oULNSnT+ITdGzwIxdv5xEi0KxPh493mqtHWk8ZReGFe3pvGUXlRqXVc+9DYzOY2Yl0EKS1ZKOmmxScQUuB/90a4zOPRuTp3h7TGytcTxsy7YdW7Ek32Xyiy+R7vP0XRaH6p3a4JpjUp0/X4SmWkZvCwQX9cfJ9GyQHxue3Ljc5rSC5Pq1jhN6UXl1nV5VKAOWy8ZTsVmtTCsbI5lw+r03DkTTX0dXhTo4bb/9BMqNq+NYRVzqnVpTP+Di/A970LAjeLv238bnrvOUm1EB2zzzhMN8s4Tr58rWu+roTQtdJ7QrWRKg7zzhK2S80StGX2wKHCecJjUg6qDWxNQ4DxRf/kIzFrUQreyORUaVafFL1+iYaCD/zHF4eZlxX33OZym98Eubx/tmLePehWow04/TqJ5gTp0332eym0daZS3jzZ6g300s9A+qq6rRculw7FsbI9BJTNsmtfm09/mkBabJIbrfsTEkF3hnVSvXp1Hjx6xbt06Fi9eTFBQEFpaWtSpU4d58+YxdepUJBIJZ86cYcmSJYwfP57IyEisrKxo27Ztsc8/Umbjxo0kJSXRp08fDAwMmDt3LvHxikNGfvvtN9asWcPcuXMJDg7G1NSUFi1a0LNnz2K2WvZCTtxF00SfmnMGoGVhTKJHEHdHfktqUO5VZ21LY3Qq5t/TmhIQyd2R31Jv5WjsxnUhLTyWJ0t/lz+DFCAtJIbbQ7+h3qpRdLjyDWlhsfj+cg6vrSflaezG5k721PpvxQkCHn65k8Cj11UZ8ht76uHF+BkL5a+/3ZL7bLG+PTqzdunc8ipWEYEn76Jlok/dOf3RtjAm/mUQN0ZtJCWvDnUsjNEtUIfJgZFcH7WRRitHYT+2C6nhsTxatk/+DFIAqa4WTuvHoWNdgey0DBK9Q7g7fQeBJ/N7/e3z6rDj8WUK5bn35S5e/fF+dfh0+2nUtTVpvm4sWka6RD7y4cKIDQrPINW3MVMYFh7p4sW1qVtpvGAwjeYPItE/HOcpWxWew/nq5D20TAxoOLs/OhbGxL4M4tLojSQreQ5fSYyqW+O0eAiaxvokBUXivvkkz39+s/uhXpehfl4Z4l4GcblAGXQsjdGzyR+NkRQYyZXR39FkxShqftaZlPBYHizfJ38G6evYr0/dSqMFg2mYF/v1QrHrWlegzbZpaFUwID06gciH3pzt/bX8c3NycjCsbk37n7/MTRObRPRjX84NWEN8KQ+VfxP3d+TWaee1Y9E21CXUzYc/R25QeDagoY2ZQm9DiKsXp6dvpdW8wbSaO4g4/3BOT9sqf0YnwOXl+2g9bxCd14xFx8yQ5PBYHh+8wp2f/pbHVaG6NXUHfYmOiQFpcUmEPfblyKA1RJdBXOUdn4F1Bey75s42+9n5dQplOjpkLYF3lQ+dfBcP82Jstyb3exnu5sOJQjEaVDRTuG0lzNWL89O20nz+YJrNG0S8fzjnp24t8gzS0viec8H5qz04TetD21VjiPUJ5eyknwgtZjTPu3DNi6/D2rFoGeoS5ubDP4XjK1SHoa5enJ2+lRbzBtNibm58Z6cpxqdvXYHuW6ehY2JAakwCYQ+9+aPf1yQWOO7oWRjTdtlIdM1yh32/+Osm9zcrnwTwXQTlnSfq5J0nEko5T6QERnJj1EYarhxF9bG55/pHy/bJn0EKueeJxuvHoZt3nkjwDuHe9B0EFThP6FhXoPn26fLjTvRDby73+lr+uWXtUV4dti2wj54qVIf6SvbRC9O20mz+YD6ZN4gE/3AuTt1a5BmkJZHl5FChVmVqDGyNlqEeKRFxBN95zoWpW4t9/ul/XjlOavS/QiJT5UNlBOEjcMJqRHkXQaV6Pl1T3kVQqeOOy0pP9D8utbQhlf/j/j8M5Yko2w444QPT/n/wSyr74z7MYJP5cVdipPpHXoHA1MAD5V0EpVJPflem29PpM69Mt/df8P/hPC8IgiAIgiAIgiD8B4khu4IgCIIgCIIgCKoghuyWSjRIBUEQBEEQBEEQVOEjnRm3LIkhu4IgCIIgCIIgCEK5ED2kgiAIgiAIgiAIqiCG7JZKNEgFQRAEQRAEQRBUQQzZLZUYsisIgiAIgiAIgiCUC9FDKgiCIAiCIAiCoAqih7RUokEqCIIgCIIgCIKgCjJZeZfgP08M2RUEQRAEQRAEQRDKheghFQRBEARBEARBUAUxZLdUokEqCIIgCIIgCIKgCqJBWirRIBWEUgyMuVbeRVCpg47LyrsIKjXgyeryLoLKZWxdUt5FUKmYfyPKuwgq9yzAvLyLoFKndT7ue6g6p0nLuwgqF67+cd/l1cTi4z7OxMfplHcRBKFYokEqCIIgCIIgCIKgCjLRQ1oa0SAVBEEQBEEQBEFQBTFkt1Qf9/gLQRAEQRAEQRAE4T9L9JAKgiAIgiAIgiCogngOaalEg1QQBEEQBEEQBEEVxJDdUokhu4IgCIIgCIIgCEK5ED2kgiAIgiAIgiAIqiB6SEslGqSCIAiCIAiCIAiqIB77UioxZFcQBEEQBEEQBEEoF6KHVBAEQRAEQRAEQQVkOWKW3dKIHtL/R2xtbdm0adMbp9+7dy/GxsYqK48gCIIgCIIgfNRycsp2+QiJHtIPaOzYsfz+++9MmjSJnTt3Krw3depUduzYwWeffcbevXvf63P27t3LrFmziIuLU1j/4MED9PT03mvbb0MikfD333/Tr1+/t8pna2vLrFmzmDVrlnxdcTH9r1i+bA4TJ4zExMSI+/cfMePLJTx/7lliHiMjQ1avWkj/fj0wMTHC71UgCxas4uy5K/I0NjZWrF/3Fd27dURHRxtPL1+++GIuDx89UUkc9p91pubUT9GxMCbeM5hHy/cTde9lsenNW9Si4YpRGNWoSGp4HB7bT+Oz77L8/Yo9m1BnZl/0bS1R05CS6BvOy11n8P/zpjxN7Rl9qNSzCQb2NmSnZRDl4oX7miMk+oSqJMZ35eL2hN8O/clzD28io2P4af0yOrVtWd7FKpV60y5otOyFxMCYnIggMs7tIydAeZ2q2dZGZ+zyIutTts5FFhWSv0JbF82OQ5HWbopERw9ZbCQZFw6Q7eWmoijejv6gPhiMGoLUzJRM31fE/rCdDDfl3xnNBvUwnvE56lWrINHWIjssnKTjp0k6/NcHLnW+ymO7YDetN1oWxiS9DMJj2T5i73kUm96kRW1qrRyNfs1KpIfH4rf1FIH7LimkqfpFD6p81gXtimZkxCQSfvoenmsPk5OeKU+jZWVCzWUjMOvYEKm2Jsm+oTydvYsEdz+VxVpQz1mDaDW8EzpG+vi7eXF02R7CvIKKTd9yWEc+GdAWm5qVAQh44sepjYfxf+wjT1P9k9p0/qI3VRztMLKswM9fbMT9gotK47Ab2xn7qb3QtjAm8WUwT5bvI7qE46hpi1o4rhiNQc2KpIXH4bXtFK8KHEerDG1L458mF8l3supnCvX3msOMPtRdMgyfn8/yZPn+sglKCac5A6g9ogNaxnpEPPLh5pK9xHoGl5jHrmdTms4bhGFVCxL8I7j/7TFencuvjzqjO1FnTCcMKpkDEOsZhOumvwm86g6AmrqUpgsGUbljQwyrmJORkErwzafcW3+UlPA4lcUKYDCkN0ZjB+ceV3xeEf3tDtIfPVWaVrdTawwH90KzZnUkmhpk+PgTt3M/qbfzY7X69Tt0mjYokjfl+j3CZyxVWRzFMR3dA4tJA9AwNyHNK4Dglb+S/OC50rTqFiZUXDoenXrV0bKzIeq30wSv+rVQIimWUwdRYVBHNCxNSfcNJuSb30m89vADRCP814gG6QdWuXJljhw5wo8//oiOjg4AaWlpHD58mCpVqrz39jMzi558XjM3N3/v7Qtvb/68qcz68gvGT5yNl5cvXy3+knNnDlOnXluSkpKV5tHQ0ODc2cNERkQzdNgXBAWHUrmSDYkF0hsbG3Hd+R+cr92mV+9RRERGUb2aLXHxCSqJo3Kf5jRcNZqHi38j8oEn9qM70vbgAs61W0BKcHSR9HqVzWl7YD6+B69yd/p2zJvWoPH6caRHJxD07wMAMmKTef7TCRK8Q8jJyMKmSyM++fEL0qPiCXPObSCYt6iF12+XiHHzQU1diuOiIbQ7soizbReQnZqukljfRWpqGjXtq9GvZ1dmL1lT3sV5I9K6zdHsPoaMf/eQHfASjSad0R61iNRt85DFF63T11K2zIb0VPlrWXKBfU4qRXv0V8iSE0j/YxOyhBgkRqYK6cuTTpf2GM+ZSuyGzaQ/for+gF6Y/7SesCHjyQ6PKJJelppG0h//kOHtiyw1Da2GjpgsnoUsLY3kv//94OW36tuC2qs/4/mi3cTef0nlMZ1xOryIm23mkqbke6hTxRynQwsJOnAF92lbMfmkJnW+mUBGdALh/94HwHpgK2osGc7T2buIe+CJbjVrHDfnNnA8lu8DQN1Ij+anVhF96xmuI74hIyoBHVtLMuNTPkjcnSf3ocOETzkwbwcRfqF0nzGAGQeWsKrjbNKT05TmcWheF9eTtzn28CVZ6Zl0ntSHafuXsLbLXOLDYwHQ0tUi+IU/d4858/muuSqPo2Lf5jiuGsPjRXuIfuCJ3ehOtDi0kMtt55OqpP50q5jT4uAC/A9cxWX6Nkyb1qDBN+PJiE4gJO84CpCZkMKlVorlV9YYNW5YDdvRHYl/5l/2wRXQYGov6n/eA+c5u4jzDaPxzL58emgRR9vNJ7OY+rJsbE/n7dN5sPFPXp1zwbZ7EzrvmM7JAauJeJR7ESE5NIZ764+S4BcOQI3Bbei2ew5/dV9CrGcw6jqamNWz5eGmf4h+HoCWsS4tV4ym+545HP+06MW0sqLXrR2mC6YQtXYL6W7PMBj0KVbb1xHUfwLZYZFF0ms3diT17kNituwhJzEZg77dsNy8ipBRM8jwyI01Ys5KJBr5P9PVjA2p+Mcuki9eV1kcxTHu1ZqKyycStGwnyS4vMBvRnWq/f41H52lkhkQVSa+mqUFWdDzhW49hPrGv0m1azxuFSf/2BC7aSrp3EAbtGmP382K8Biwk9ZmviiP6wMSkRqUSQ3Y/sMaNG1OlShWOHz8uX3f8+HEqV65Mo0aNFNKeO3eO1q1bY2xsjKmpKb169cLHJ//K7qtXr5BIJPzxxx+0b98ebW1tDhw4wLhx44iPj0cikSCRSFixYgVQdMjuDz/8gKOjI3p6elSuXJmpU6eSlJT0xrFkZGQwffp0rK2t0dbWxtbWlvXr18s/C6B///5IJBL5ax8fH/r27YulpSX6+vo0bdqUS5fyr9S3b98ef39/Zs+eLS+/s7NzsTFJJBL++ecfhXIZGxvLe5lLKuOHMnPGRNZ/s5l//jnLs2cvGTd+Frq6Ogwf1r/YPOPGDqOCiTEDBo3n9h0XAgKCuXX7Ae7u+VcjF8yfSlBQCBM/n8MDFzf8/YO4cvUmvr6q+aFRc1IP/A4743vImUSvEB4tP0BqSDTVP+usNH31MZ1ICY7m0fIDJHqF4HvIGb8j16g5+VN5msg7Lwg+60KiVwjJ/hF4/Xqe+BcBmH1SU57m+ohvefXHdRI8g4l7HsD92bvQq2RGhQZ2KonzXbVp0ZSZX3xGl/atyrsob0yjxadkPbxK1sOryKJCyDi3D1l8NOpNupSYT5acgCwpXr4gy78/Rr1RByQ6+qQf+Z6cQE9k8VHkBLwkJzxA1eG8EYMRg0g+cZbkE2fIehVA3A/byQ6PQH9Qb6XpMz29SblwlSxff7JDw0k5e4m0uy5oNXT8wCXPZTv5U4IOXSXo4FWSvULwWLaPtOBoqoxVXmeVx3QhLSgaj2X7SPYKIejgVYIOX8Vuai95GmOnGsQ98CT0+C1SAyOJvuZO6N+3MWxQTZ6m2ow+pIZE83TWTuIf+ZAaGEnMjaek+oerPGaADuN7cn7b3zw+f59Qz0D2z92Gho4WTfq2LjbP77O2cOPABYKf+xPuE8KhRbuQSCTUbJVfd8+d3Tj9/VEen7//IcKg+qSe+B92xv+QM0leITxZvp/U4GjsijmO2o7pRGpQNE+W7yfJKwT/Q874H3bGfkovxYQyGemR8QpLYVJdLZpsm4bb3F/JjFd+MbSsOE7ozsMtJ/A760LsyyCuzt6Fuo4m9v2KHzXiOLE7QTee4rbtFHE+obhtO0XIrec4TuguT+N/6RGBVx4T7xdGvF8YD749RmZKGhaN7QHISEzl3xEb8D19j3jfUCIe+nBr2T7MG1RD38ZUZfEajh5I4t/nSPr7LJl+AcRs3EFWWCSGQ5QfV2I27iB+7x9kPPMkKyCY2C17yAwIRrddC3manIREsqNj5YtO88a5F8LKoUFqPrEvMUcvEXPkIuneQQSv+pXM0CjMRvVUmj4jKILglb8Se/wqOQnK97UKA9oTse0YiVddyQgMJ/rAWRKvPcL8834qjKSc5MjKdvkIiQZpORg3bhy//fab/PWePXsYP358kXTJycnMmTOHBw8ecPnyZdTU1Ojfvz85hcaPL1y4kJkzZ/LixQs6derEpk2bMDQ0JDQ0lNDQUObNm6e0HGpqamzevJmnT5/y+++/c+XKFRYsWPDGcWzevJmTJ0/yxx9/8PLlSw4cOCBveD54kHvl9rfffiM0NFT+OikpiZ49e3Lp0iUePXpEt27d6N27NwEBuT9Wjx8/TqVKlVi1apW8/C1btnzjmN6mjB+CnV0VrK0tuXjpmnxdRkYG12/cpUWLJsXm692rC3fvubJl81qCA91we3SZRQtnoKaW/5Xt1asrrq7uHDm8i5Cgxzy4f54J40eoJA41DSkm9e0Iu6Y4rDHs2hPMmjgozWPaxKFoemd3KjSwQ6IuVZrHonVdDKpbE3m3+OGHGga6AGTEvvnFE0EJqRQ1GzuyfdwVVmf7uCOtXKPErDqT1qMzdzvaY5agZltHcbM1G5MT5IXmp+PQnbcTnanfotGmL0gkZR7CW1NXR7NWDdLuKQ7JTLvnimb9um+0CY0a9mjVr0v6Q/fSE5cxiYYUw/p2RDkrfnbUNXeMmyivM+MmDkRdK5T+qjuGDarJv4ex9z0wrG+HUaPqAOhUtcC8UyMiL+UPnbPo6kTCY18a/jKLDs920fLSeiqN6liW4RXLtLIFRhYmeNzIjyMrIwvve8+p5lTyvlqQpo4WUg11UuLK59gh0ZBiXN+OiEL1F3HtCRWaKo+jgpMDEYWOoxHO7hgXOo5K9bTp6vIT3R5uofn+eRjVq1pkWw2+GUfYpUdE3lA+jLSsGFQxR8/SmKAC5c7JyCL0rgeWxZwvACyc7BXyAAQ6uxebR6ImoXqf5mjoaBHu6lXsdjUNdJDl5JCeoKLefHV1tGrXIPWOq8Lq1DuuaDV4s+MKEglqurpkxycWm8Sgfw+SzjkjS1Xew6wqEg11dB3tSbzxSGF94vVH6DnVevftamoU6cXPSctAv0ntd96m8L9LDNktB6NHj2bx4sXyHs5bt25x5MgRnJ2dFdINHDhQ4fXu3buxsLDg+fPn1KtXT75+1qxZDBgwQP7ayMgIiUSClZVVieUoeI+mnZ0dq1evZsqUKWzfvv2N4ggICMDBwYHWrVsjkUioWjX/BPh6eLCxsbFCORo0aECDBvn3RKxZs4a///6bkydPMn36dCpUqIBUKsXAwEAh35vG9DZl/BCsLC0ACA9XHNISHh5J1SqVis1nV60qHaq24tDhv+ndZzT2DnZs+Wkd6upS1qzdBEA1uypMmjSaTT/9wjcbNtO0SSM2/biK9IwMDhz4s0zj0KxggJq6lLRCV93TIuPRNjdSmkfH3IgwJenVNNTRqmBAWkQcABoGOvR+tBWppjqy7BxcF+8l/HrxP5garhhJ5D0P4l8Wf++YUDqJriESNSmyZMU6kiXHI9FXXqeyxDjST/5CTqgvSDVQb9AG7c+WkLZ3NTn+uRcR1EwskNiZk+V+i7SDG5CYWqHVcxyoScm8dlzpdj8UNWMjJOpSsmNiFdZnR8eibVqhxLzWp48gNTECqZSEX/aRfOKMKouqlGYFQ9TUpWQU+l5lRMajZWGsNI+WhbHS9Goa6mhWMCA9Io6wf+6gaWpIs5MrQQJqGuoE/HYBvy0n5Xl0qlpQ+bPOvNp1Bp+f/sG4UXVqrxlLTnomIcdulHmsBRmaGwOQWCiOxMh4KlR681tR+i4cQXxYDB63VHOPfWm08o6jhXsv0yPj0SrmOKptYUxEpHuR9AXrL9ErhIdf7iThRSAaBjpU+7w7bU6u4GqnxST7hQFQsW8LjBxtudZ9mUpiK0g3r75SoxTjTI2KR7+iWYn5lOXRLfS/qVCrEv1OrECqpUFmchrnP99EnFcIyki1NGi2eCje/9whM0k1tw1ITfKOK9FFjytSM5M32obRmEFIdLRJvnBN6fua9Wqi6WBH5Irv37u8b0tqYohEXUpmVJzC+syoeAzy6vpdJF5/hPnEviTde0qGfxj6rRpg1LUZqH2EfWUf6UREZUk0SMuBmZkZn376Kb///jsymYxPP/0UM7OiB2kfHx+WLVvG3bt3iYqKkveMBgQEKDRImzQpvqetJFevXmXdunU8f/6chIQEsrKySEtLIzk5+Y0mPxo7dixdunShZs2adO/enV69etG1a9cS8yQnJ7Ny5UpOnz5NSEgIWVlZpKamyntIy9rbljE9PZ30dMX7EmUyGZI37N0ZPrw/O7ZtkL/u03eMfBsFSSSSIusKUlNTIyIimslTFpCTk8PDR0+wsbZi7pzJ8gapmpoarq7uLF32DQBubs+oU6cGk78YU+YNUrnCZS7t/1IkuaTIdjKT0rjQ+SvU9bSxbF2XhitGkuQfQeSdF0U213jdWIzrVOFy31XvUnpBmSK7oUTZytyk0aFkRedPJpUR5IXEsAIaLXuRntcgRaKGLDmBjFO/5NZzqB8ZBiZotOxV7g1SucLhSSi6bxcS8cUs1HR00HSsjdG0z8kKDCblwlWVFbFkhb9YRY8xismLps9dnbu+Qss6VJ/Vn+eLdhP30BtdWytqr/mM6uFx+PyYW2cSNTXiH/vite4IAIlPX6FfqxJVxnYp8wZpk76tGb7uc/nrHeO/UShvfhwlH0cL6jypD059WvHTsJVkKbm38kMqUuTiv3J56Ys57uatj33oTexDb/nb0fc96XBxLdUmdOXJ0n3o2FTAcc0Ybg9dr/S+0vdl378lbb/JH+V19rPv8spXKOEbnEeL/m+K1nGcTyh/dluCpqEu1Xo2pcOPkzg5aE2RRqmaupRO26aBmho3vtr7htG8B2X19Ab7p173DhhPGU34l1+TExOnNI1B/+5kePmR8bT4ya9UrsjvGErcb0sTtOIXqnwzndpXtoMM0v1DiT52CdPByoevCx830SAtJ+PHj2f69OkAbNu2TWma3r17U7lyZX755RdsbGzIycmhXr16ZGRkKKR7l5lz/f396dmzJ5MnT2b16tVUqFCBmzdvMmHChBInRiqocePG+Pn5cfbsWS5dusSQIUPo3Lkzf/5ZfGNo/vz5nD9/nu+++w57e3t0dHQYNGhQkZjelLKGXcHyv20Z169fz8qVKxU/Q00fidTwjcpz6tQF7t/PH9aipaUJgJWVOWFh+ROmWFiYER5RdCKA18JCw8nMzFIYnu3h4YW1tSUaGhpkZmYSGhrB8xeKM/V6eHgzoL/yezreR0ZMIjlZ2WgX6oXRNjMs0mv6WmpkPNoWile2tcwMycnMIr3gcFuZjKRXufehxT3zx9DBhtoz+xRpkDZeM4aKXRtzpf9qUkNj3j+o/+dkKQnIcrKL9IZK9AyRJb35xFg5Qd6o18+/j0+WGAc5WQo/XmSRwagZmIBUCtnZ7132d5UTF48sKxupqWKvhbSCSZFe08KyQ8LIBjJ9/JBWMMHwi88+eIM0IyaBnKxsNAv1SmiaGRXpBX0tPSIOTYui6XMys8jM+x7aLxxCyLEbBB3MjSfpRSBSXS3qffc5Ppv+zr0/MTyWJE/FUQlJniFYftqsbIIr4MklF1655Q/BVNfUAMDQwpiEyDj5egMzQxKjlMddUKfPe9F1Wj+2jlxDiEf53cucLj+OFj4uGpFeTBxpEXFFjruvj6PF3rYgkxHr5ot+tdwRRcb1q6FtbkT7C2vlSdTUpZg2r4Xd+K6crDLmve5L87/wkD8f5c9vIdXM/WmpY25ESt5IGAAdU0NSitlPAVIi44r0huqYGpIapXg8ysnMJiHvnBHl7od5g2o4TujOjUV7FOLrvHMGhlXMOTVkvcp6RwGyY/OOK2aKoyykFYzJjo4rMa9et3aYrZhDxPzVpN17pDSNRFsL/W4diN3+e1kV+a1kxyYgy8pGw1zxuKluakRWoV7Tt9puTAJ+X6xDoqWBurEBmeExWC/6jPTAD3Nf+gclekhL9RH2i/9v6N69OxkZGWRkZNCtW7ci70dHR/PixQuWLl1Kp06dqF27NrGxJf9gek1TU5PsUn70ubi4kJWVxffff0/z5s2pUaMGISHKh7yUxNDQkKFDh/LLL79w9OhR/vrrL2JichsLGhoaRcpx48YNxo4dS//+/XF0dMTKyopXr16VWv7iYjI3Nyc0NL/HxsvLi5QUxftESipjYYsXLyY+Pl5hkagZvPH/IykpGR+fV/Ll+XNPQkPD6dyprTyNhoYGbds0586d4h8tcPuOC9Wr2yr0zDo4VCMkJEze4L595wE1a1RXyFfDoRoBASVPq/8ucjKziXX3w6ptPYX1lm0diXJRfu9OtIsXlm0VJ36xaudIzGM/ZFkl7J8SifwHzWuN135GxZ5NuTp4LcmBRWcsFN5BdjY5IX5Iq9dXWC2t7kh2YMmPJCpIzdoWWVJc/mYDXyKpYKXQGyIxtSYnMbZcG6MAZGWR4eGJdjMnhdXanziR4f7szbcjkSDR0CjjwpVOlplNgrsfZu0Uv1dmbR2Jc1FeZ3EuXpgV+h6ata9PwmNf+fdQqqNZ9MHtOTm5dZhXjbEPPNGrbqOQRK+6NalBxV9Ye1fpyWlE+YfLlzCvIOIjYqnVOn9flWpIsW9WB1/XkvfVTl/0pvuMgWz/bD0BT8p35k5ZZjZx7n6YF6o/83b1iHmgPI4YVy/M2ykedy3a1yeulOOoUb2qpOU95iTyxlMut1/A1c6L5Uusmw9Bf93iaufF7z1JSmZyGgmvwuVLrGcwyeFxVCpwvlDTkGLdvBbhxZwvACJcvRXyAFRq51hiHsi9MF3wnPG6MWpka8npYd+Qrup7hrOySH/hiU7zxgqrdZo3Jv1x8ccVve4dMFs1n8jF60m9UfykWnpd24GmBkn/Xio2jSrJMrNIeeKNQZuGCusN2jQk2bX4+R7eePvpmWSGx4C6FOMeLUm4cO+9t/mfI5OV7fKWtm/fjp2dHdra2jg5OXHjRsmjWtLT01myZAlVq1ZFS0uL6tWrs2fPnhLzvC/RQ1pOpFIpL168kP9dmImJCaampvz8889YW1sTEBDAokWL3mjbtra2JCUlcfnyZRo0aICuri66uroKaapXr05WVhZbtmyhd+/e3Lp1q8izUUvz448/Ym1tTcOGDVFTU+PYsWNYWVlhbGwsL8fly5dp1aoVWlpamJiYYG9vz/Hjx+nduzcSiYRly5YVmaTJ1taW69evM2zYMLS0tDAzMys2po4dO7J161aaN29OTk4OCxcuRKPAD8XSyliYlpYWWlpaCuvedLhucTZv+ZVFC2fg5e2Ht7cfixbOICUllcNH/pan+W3PT4SEhLJkae7QtJ279jFt6jh+/GEV27b/hoO9HYsWzmDrtvwDwk8//cKN6ydYtHAGx/48RdOmDZk4cSSTp775xFRv4+WuszTbMoWYx35EuXpRfVRHdCuayp8r6vjVUHStTLg3M3c/8tl3GYfxXWi4YiQ+B69i5uSA3fD23J26Vb7N2jP6EPPYl6RX4ahpqmPdqSG2g1vjuih/0i+n9WOp0r8lN8f9QFZSmvye1czEFLLTynfoXUEpKakEBOVf1AkOCcfD0wcjQwOsrSzKsWTFy7zzL1oDppET4kt2oCcaTp2QGJmR5ZL7w0ej0zAkhiZk/L0DAPXmPZDFRZITEQRSddTrt0a9TjPSjv4g32bWg4tofNINze6fkXn/HGoVrNFs04/Me+fKJcbCEg/9ienKRWQ89yT9yXP0+3+K1MqCpL9OAWA0bQJSczNiVuQOvdcf3JessAiyXuX2rGk1rIfBqMEkHv2nXMr/aue/1N86jfjHvsS5eFJ5dGe0K5kR8HtundVYMgwtqwo8mZE7F0DgvotUmdCVWitHE3jgMsZNalBpRAceT94s32bkhYfYTu5JwlM/4vOG7NovHELEBVd5Y+XVrn9pfnoV1b7sR9iJOxg1tqfS6I48m/fLB4n76p4zdJ3Wj4hXoUT6hdFtWj8yU9NxOZH/zOLR308jPjyGk98eBnKH6X46Zwi/f7mZ6KAIDPKOHenJaWSk5N6aoamrhblt/twEppUtqFinKilxScSGFP/oo3fls+sMTlumEvfYlxgXL2xHdUS3ohl+ecfROl8NRdu6Ag9n5H7nXu27TLXxXam3YhSvDl6hQhMHqg5vj8uULfJt1pw7gFhXb5J8w3LvIZ3YDaO6VXm8eC8AWclpJHoo9m5np6STEZtUZH1ZebL7HI2m9yHeL5x4vzAazehDVmoG3v/clqfpsGkSyWGx3P/mj7w85+nz11IaTO2F/3lXqnZzomLrupwcsFqe55OFQwi4+pikkGg09bWp3qcF1i1qc2bUtwBIpGp02TUTM0dbzn72PRKpGjqv6z0uiZxM1VwUS9j/F+ZrF5L+3JP0xy8wGNgTdWsLEo+dBsBk5nikFmZELc0tp173DpivWUD0t9tJd38hH7WRk56OLEnxorpB/+6kXL1FTgkTHqla5K8nqPLjbFLcvUl+6IHp8G5o2JgTdfAsANYLxqBhVYGAOZvkeXTq5M6Er6anjdTUEJ06drkjpLwCAdBtWAMNK1NSn/miYWWK1ezhoCYhYtd/5NaOj8TRo0eZNWsW27dvp1WrVuzatYsePXrw/PnzYh83OWTIEMLDw9m9ezf29vZERESQlZWl0nKKBmk5MjQsfhiompoaR44cYebMmdSrV4+aNWuyefNm2rdvX+p2W7ZsyeTJkxk6dCjR0dF8/fXX8sekvNawYUN++OEHNmzYwOLFi2nbti3r169nzJgxb1x+fX19NmzYgJeXF1KplKZNm3LmzBn5TLDff/89c+bM4ZdffqFixYq8evWKH3/8kfHjx9OyZUvMzMxYuHAhCQmKw3FWrVrFpEmTqF69Ounp6chksmJj+v777xk3bhxt27bFxsaGn376CVdX1zcu44ew8bvt6Ohos3XzOkxMjLh//xE9Ph2h8AzSKpVtFBrmQUEh9Og5gu+/W8Ej14sEB4exZetuvt2YP7zbxfUxgwZPZM2aRSxdMgu/V4HMmfs1hw//jSoEnryLlok+def0R9vCmPiXQdwYtZGUvB4SHQtjdCvmT6ufHBjJ9VEbabRyFPZju5AaHsujZfvkzyCF3McQOK0fh451BbLTMkj0DuHu9B0EnrwrT2Of9ziLjscVJ+O49+UuXv3x4ae/L85TDy/Gz1gof/3tlp8B6NujM2uXqv75hu8i+9ldMnQN0Gg3AE19Y3IiAkk7uAFZfG6dSgyMUTPKv79dIlVHo+tIJAYVICuDnIgg0g5uINvLTZ5GlhBD2v71aHYfjc6UDcgSYsm8d5bMmycLf3y5SL3oTJyRIYYTRyM1q0CmzyuiZi0mO29IvdTMFGnBCwgSCcbTJiC1sYLsbLKCQonb+ivJx0+XS/nDTtxBw0Qf+zkD0bI0JtEjENcR35CW9z3UsjBBp8DEMakBkbiO2ECtVWOoMq4raeGxvFiyV/4MUgCfH48jk8lwWDQUbasKZEQnEHHBFa/1R+VpEtx8eTTuB2osGUb1OQNIDYjEY9k+Qv+69UHivrTzJJramgxdPQFdIz1euXmzdfQ6hWeQVqhoiqzA8/7ajO6ChpYGE3cqfv/ObDrGmU25t21UrV+dL498LX9v4LLPALj7pzMH5u0o8ziCT9xF00SfWnMGoGVhTKJHEHdGfivvada2VDyOpgREcmfktziuHI3duC6khcfivvR3hWeQahjq0vC7CWiZG5OVmELcE39u9FtNXIFhtB/a4+2nUdfWpPXasWgZ6RLh5sO/IzcoPINUv6KZQs98uKsXl6Ztpen8wTSdN4gE/3AuT90qfwYpgI65IR1/moyuhTEZiSlEvwjkzKhvCc6bOVjPugK23XJHQAy+uE6hTCcHryVUydwEZSH5/DXUjAwx/mIU6uYVyPB+Rfi0JWSF5h9X1AscVwwGfYpEQx2zJTNhyUz5+sQTF4havlH+Wr1qRbQbOxI6Kf/cUh7iTt9EamKA1cyhqFtUIM3TH9+xq8gMzh2xpGFhgqaN4gRjNc/+JP9bt74DFfq1JyMwnOetc+8Pl2hpYD1vJJqVrchJSSPhqgv+s34ku5jHxPxPK8chuz/88AMTJkxg4sSJAGzatInz58+zY8cOpY9BPHfuHNeuXcPX15cKFXKHoX+Ip1NIZG86I4Ag/D+lrlmxvIugUgdN25d3EVRqwJPVpSf6H5exdUl5F0GlYv6NKD3R/7hnAW8+W+z/otM6H/dPjc5pyh9l9TEJV/+47/LqavoR3rtYQHycTnkXQeUa+v83LnwWlvLdxDLdnnTGtiITcCob4ZeRkYGuri7Hjh2jf//+8vVffvklbm5uXLtWdFbnqVOn4unpSZMmTdi/fz96enr06dOH1atXo6Ojun3o4z66CIIgCIIgCIIgfCTWr1+PkZGRwqKstzMqKors7GwsLS0V1ltaWhIWFqZ0276+vty8eZOnT5/y999/s2nTJv7880+mTZumklheE0N2BUEQBEEQBEEQVEFWtkN2Fy9ezJw5cxTWFe4dLajwXCglPc4wJycHiUTCwYMHMTLKvf/6hx9+YNCgQWzbtk1lvaSiQSoIgiAIgiAIgqAK7zmTdWHKhucqY2ZmhlQqLdIbGhERUaTX9DVra2sqVqwob4wC1K5dG5lMRlBQEA4ODu9X+GKIIbuCIAiCIAiCIAgfEU1NTZycnLh48aLC+osXL9KyZUuleVq1akVISAhJSfmPS/L09ERNTY1KlSqprKyiQSoIgiAIgiAIgqACspycMl3expw5c/j111/Zs2cPL168YPbs2QQEBDB58mQgd/hvwSdsjBgxAlNTU8aNG8fz58+5fv068+fPZ/z48Sqd1EgM2RUEQRAEQRAEQVCFMh6y+zZePy5x1apVhIaGUq9ePc6cOUPVqlUBCA0NJSAgQJ5eX1+fixcvMmPGDJo0aYKpqSlDhgxhzZo1Ki2naJAKgiAIgiAIgiB8hKZOncrUqVOVvrd3794i62rVqlVkmK+qiQapIAiCIAiCIAiCKpTxLLsfI9EgFQRBEARBEARBUIVyHLL7v0JMaiQIgiAIgiAIgiCUC9FDKgiCIAiCIAiCoApvOTPu/0eiQSoIgiAIgiAIgqAKYshuqUSDVBBKYaytV95FUKlUNUl5F0GlMrYuKe8iqJzm9LXlXQSVMjX5tryLoHKW34eWdxFUqma2QXkXQaUMZRnlXQSVu6r+cd/l1Tn7445PTSIaRcJ/l2iQCoIgCIIgCIIgqIKYZbdUokEqCIIgCIIgCIKgCmLIbqk+7vEJgiAIgiAIgiAIwn+W6CEVBEEQBEEQBEFQAZmYZbdUokEqCIIgCIIgCIKgCmLIbqnEkF1BEARBEARBEAShXIgeUkEQBEEQBEEQBFUQPaSlEg1SQRAEQRAEQRAEVRCPfSmVGLIrCIIgCIIgCIIglAvRQyoIgiAIgiAIgqAKYshuqUQPqfA/w9bWlk2bNpV3MQRBEARBEAThjchyZGW6fIxED+lHJDAwkBUrVnD27FmioqKwtramX79+LF++HFNT0/Iu3hvbu3cvs2bNIi4uTmH9gwcP0NPTK59CvacFi2cwZuwQjIyNeOjymAVzV/LSw7vY9MNG9Gfrzg1F1lc0r0d6egYAUqmUBV/NYNDg3lhYmhMeFsmRQ8f5/tvtyGRld8BqOGcANUZ2QNNIj6hHPtxdspc4z+AS81Tt2ZRG8wdhUNWCRP8IHm44RsA5F4U0NT/rTL3JPdG1MCbWM5j7Xx8g4v5LpdtrsWE8NUd15P7X+3n+63n5eoOqFjRZNgLLT2qgpqlBsLM795b+TlpUwvsHnke9aRc0WvZCYmBMTkQQGef2kROgvJxqtrXRGbu8yPqUrXORRYXkr9DWRbPjUKS1myLR0UMWG0nGhQNke7mVWbnLmovbE3479CfPPbyJjI7hp/XL6NS2ZXkX642o12+HeuMuSPSMkEWHkHH9GDkhxX//kKqj8cmnSGt9gkTXEFlSHJkPzpL9/Hbu23Vbo167GWqmNgDkRASQefsEOeGvPkA0RZmN7oHFpP5oWJiQ5hVA0MrdJN9/rjStuoUJFZeOQ9fRHi07ayJ/O03wyt1F0plP6I3ZqB5oVjQjKyaRuDO3CdmwD1l6pqrDAeCT2QOoO7ID2kZ6hD3y4drSvcSUctyp3qMpzecNwqiqBfH+EdzZeAzfAsedT2YPoNmcAQp5kiPi2OM0Xf56RuABpdu+ueYwj3b9+x4Rlazi2K5UmdYHTQtjkl8G4bVsL/H3PJSm1bQwxn7lGAzqV0O3mhVBv57Fa9nvCmn0albCbsFQDOrboVPFAs9lewn6+YzKyv+mus8aRMvhHdEx0sffzZs/l+0hzCuo2PRWDpXoOWcwlRyrYVrJnOOrfufanrMKabT0tOk5dwj1uzZF38yI4GevOL5yLwHuvqoOR4Hh0F4YjxuM1LwCmd7+RG3YSdrDp0rT6nVuheHQXmjVrIZEU4MMb39ith8g9barPI1B3y5YrJ1XJK9v417IMj7M97AkpqN6Yj5pAOoWJqR5BhCy6hdSHig/7hh2a4HpqB7o1MmNN80rgPBNh0i6/ugDl1r4rxI9pB8JX19fmjRpgqenJ4cPH8bb25udO3dy+fJlWrRoQUxMTHkX8b2Zm5ujq6tb3sV4azNmfc6UaeNYOG81XdoPJCIiir9O/Ia+fsmN64T4ROrYt1RYXjdGAWbO/pyx44ezaP5qWjbtwcrl3zJ95gQ+nzy6zMpeb2ov6nzRg7tLf+f0p8tJjYyj6+FFqOtpF5vH3Mmedjum4/PXTU52+Qqfv27Sfud0zBpVl6ex7dOMT1aMwn3zSU52W0rE/Zd0OTAfPZuiF06qdHPCvFF1kkMV92F1HS26HFoIMhnnhqzjTL+VSDWkdNo7FySSMolfWrc5mt3HkHnjH1J3LiYn4CXaoxYhMSr5Ak/KltmkfDdZvsiiQwtsVIr26K+QGJuT/scmUrfMJf3UL8gS/tvf0dTUNGraV+OrOVPLuyhvRerghEbbwWQ+OEvaobVkh3ij1Xc6EgOTYvNo9vgctco1ybi0n7R9K8g4txtZbFj+NivVINvThbS/fiTtj2+RJcag1X8mEj1j1QdUiHHv1lT8egLhW4/h0XM2SfefU/335WjYmClNr6apQVZMAuFbj5H6/JXSNCb92mGzcAxhm47wouN0AuZvwaR3a2wWjlFhJPkaT+lFo897cH3p7xzttZyUyDj6HlqERgnHHavG9nTfPh2P4zc53O0rPI7fpPv26Vg2rK6QLvplILsbT5Mvh7osVni/4Hu7G0/j0tyfkeXk4HP2vkpiBbDo2wKH1WN5tek4DzovJP7eCxoc/gqtisqPM2paGmRGJ+C/6ThJz/yVp9HRItU/HJ+1h0gPj1VZ2d9Gp8l96DChJ38u/40f+nxFYmQcUw98hVYJ9aqpo0lUQASnNhwiPkJ5HMM2TKJma0cOzNnGhm7z8bjhztQDSzGyLP47Xtb0urfDbNFkYn85TNDgqaQ+fIr1zjWoW5krTa/t5Ejq7YeETl1G0JDppD5wx3rbSjRrKe6v2YnJvGo3TGH5LzRGjXq1xnr5RCK2/oFXzy9JfvAMu70r0LBRHq9es7ok3XTDb9xKvHrPIvmOO7a/LkO7brUPXPJykiMr2+UjJBqkH4lp06ahqanJhQsXaNeuHVWqVKFHjx5cunSJ4OBglixZAkB6ejoLFiygcuXKaGlp4eDgwO7d+VfHnz17xqeffoqhoSEGBga0adMGHx8fANq3b8+sWbMUPrdfv36MHTtW/trW1pbVq1czYsQI9PX1sbGxYcuWLQp5fvjhBxwdHdHT06Ny5cpMnTqVpKQkAJydnRk3bhzx8fFIJBIkEgkrVqyQb7vgkN2AgAD69u2Lvr4+hoaGDBkyhPDwcPn7K1asoGHDhuzfvx9bW1uMjIwYNmwYiYmJ7/vvfiuTp37GD9/t4N9TF/B44cW0SQvQ0dFh4OBeJeaTyWREREQpLAU1/aQRZ/+9xMXzzgQGBHPqxHmuXrlFw0aOZVb2OhO74775BAFnXYh7GcSNWbtQ19GkWv/ie8bqTOxOyPWnPNl6inifUJ5sPUXozefUmdhdnqbu5z3wOuKM12Fn4r1DuP/1AZJDoqk5ppPCtnStTGi29jOuT9+OLCtb4T2Lpg7oVzbn5uyfifMIIs4jiJtzfsa8UXWsW9cpk/g1WnxK1sOrZD28iiwqhIxz+5DFR6PepEuJ+WTJCciS4uULBXqs1Rt1QKKjT/qR78kJ9EQWH0VOwEtywgPKpMyq0qZFU2Z+8Rld2rcq76K8FfXGncl6dovsZ7eQxYaRef0YsqRY1B3bKU2vVrUO0koOpJ/YSk6gB7LEaHLCX5ETmt/bknF+D1nu15BFBSGLDSfj8gFAglrlmh8oqnwWE/sSffQS0Ucuku4dRPDK3WSGRGE2uofS9BlBEQSv+JWYv66SnZisNI1e45oku74g9sR1MoIiSLzhRuyJ6+jWr640fVlrOKE7D7acwOecCzEvg7g4exca2prU6Ff8cafhxO4E3niK67ZTxPqE4rrtFEG3ntOwwHEHICcrh5TIePmSFqN4Pij4XkpkPNW6Nibo9gsSAiJVEitA5cm9CDl0hdCDV0jxCsZr2e+kB0dRcWxXpenTAiPxWrqXsGPXyUpIUZom0c0Hn1UHiPjnNjkfqFe7NO3G9+DCtn9wP/+AUM8gDszdjoaOFk59iz+mBLj7cnL9QR6dukNWRlaR9zW0NGjQ/RNOrj+Ez30PovzDObfpT6KDImg1quTjdFkyHjOAhOPnSfzrHJm+gURv2ElWWCSGw5Sf56M37CTut2OkP/UkMyCEmJ9+I9M/BL32zRUTymRkR8cqLP8F5hP7EfvHRWKOXiDdJ4jQVb+SGRqF6Sjlx53QVb8Sues4qe5eZLwKJWzjfjJehWLY6ZMPXPJykpNTtstHSDRIPwIxMTGcP3+eqVOnoqOjo/CelZUVI0eO5OjRo8hkMsaMGcORI0fYvHkzL168YOfOnejr6wMQHBxM27Zt0dbW5sqVK7i6ujJ+/HiysoqeBEqyceNG6tevz8OHD1m8eDGzZ8/m4sWL8vfV1NTYvHkzT58+5ffff+fKlSssWLAAgJYtW7Jp0yYMDQ0JDQ0lNDSUefOKDlmRyWT069ePmJgYrl27xsWLF/Hx8WHo0KEK6Xx8fPjnn384ffo0p0+f5tq1a3zzzTdvFc/7qGpbGUsrC5yv3JSvy8jI5Pat+zRt1rjEvHr6ujx6ehX3F9c59McuHOvXVnj/7h1X2rZrQXV7WwDq1qtFsxZOXLrgXCZl169ijq6lMSHXnsjX5WRkEXbXA4smDsXmM3eyJ+T6E4V1wdfc5XnUNKSY1rcj5JriUKaQa08VtyuR0GbzZJ7u+FfpEGE1LY3ck3WBq8XZ6ZnkZOdg2bQMGgZSKWo2dmT7uCuszvZxR1q5RolZdSatR2fudrTHLEHNVrFxLK3ZmJwgLzQ/HYfuvJ3oTP0WjTZ9y6xXVyhATYqaRRVyAl4orM72f4GatfIr89JqDcgJ90ejSVe0J3yD9piVaLQeCFKN4j9HXROkUmTpyhsHqiLRUEfXsTqJ190U1ifccEPPqdY7bzfpwQt06lVHt0Hu91GziiWGHZxIuOJaSs73Z1jFHD1LYwKuKx53gu95YO1U/HHHqrG9Qh6AgGvuWBXKY2xnyTiXLYy59QPdtk3DsIryHh0AHTNDqnZsyPOjzu8WzBuQaEgxqF+NGOfHCutjrrlj1OTDX+BQFdPKFhhZmOBxI/94mp2Rhc+9F9g5lXw8LYmauhSpupSsQo3uzLQMqjV99+/AW1FXR6uOg8JwW4CU265oN3jDi6MSCRI9HbLjFS+QqOnqUOXCPqpeOoDVtlVFelDLg0RDHZ169iTeUBxum3TjEbpOtYvJVXgjEtT0dMiO+7AdBMJ/l7iH9CPg5eWFTCajdm3lB4LatWsTGxvLgwcP+OOPP7h48SKdO3cGoFq1/B9l27Ztw8jIiCNHjqChkfvjq0aNtz9RtGrVikWLFsnz37p1ix9//JEuXXKvVhbsZbWzs2P16tVMmTKF7du3o6mpiZGRERKJBCsrq2I/49KlS7i7u+Pn50flypUB2L9/P3Xr1uXBgwc0bdoUgJycHPbu3YuBgQEAo0eP5vLly6xdu/at43oXFha5w+YiI6IV1kdGRFOpik2x+by8fJkxZRHPn3liYKDPF1PG8O+FI7Rv1Qdfn9whWpt//BlDQwPuuJwjOzsbqVTK2lU/cvzPsrnPScfCGIDUqHiF9amR8ehXUj4cEEDH3JjUyKJ5dMyNANCqYICaurTodqPi5Z8J4DitFzlZObzYfR5lIl29yUpJp8mSYbiu/wOJRILTkqGoSdXQsTRWmudtSHQNkahJkSUrllOWHI9E30hpHlliHOknf8ntTZNqoN6gDdqfLSFt72py/HPvB1MzsUBiZ06W+y3SDm5AYmqFVs9xoCYl89rx9y63kE+io59bhymK9xTLUhOQ6BkqzaNmaIaajT1kZ5Jxeifo6KPZYTgSbV0yLu1XmkejVX9kSXFFGr6qJq1giERdSlZUnML6rMg4NMzffbhi3KkbqJsa4vDX+tyRKhrqRO47Q/j2v96zxKXTNTcGih53UiLjMSjhuKNrbkxK4TxR8eiZ539Xwx95c3HWLuL8QtE1M6LJzH4M+vtrDnVaRFpcUpFt1h7UhszkNHzOuhR5r6xoVDBETV1KRqFjZkZkPJoFjof/6wzy6jWxUJyJkfGYlFCvpUlPTsPP1ZOuMwcQ5h1MYlQcTn1aUbWhPZF+YaVvoAxITfK+h9FxCuuzo+OQmr3Z99B47EDUdLRJPn9Nvi7DL5CIpd+R4fUKNT1djEb1o+L+HwgaOIXMgJAStqZa8ngj4xTWZ0bGYWBm/EbbMP+8H2q6WsT9e7P0xB+Dj3SYbVkSDdL/B15PcOPn54dUKqVdO+VD1dzc3GjTpo28MfquWrRoUeR1waG2V69eZd26dTx//pyEhASysrJIS0sjOTn5jSctevHiBZUrV5Y3RgHq1KmDsbExL168kDdIbW1t5Y1RAGtrayIiIordbnp6Ounp6QrrZLIcJJI3G0wwaEhvvtu0Sv56xOAv8raheDCSSCQlTjzk+uAxrg/yr5jfu+vKlRv/MHHSaL5asAaA/gM/ZfDQPkyaMBePF17Uq1+btd98RVhYBEcP/f1G5S2oWv+WtNgwXv760pjvcv8oVEyJRFJkXRFK8xRaWeR/kr/O1NGWOhO6cbL70mI/Ij0mEedJm2m+fhy1x3dFliPD78Qdotz9kGWX4ZCWIrFKlK3MTRodSlaB+0UzgryQGFZAo2Uv0vMapEjUkCUnkHHql9x4Q/3IMDBBo2Uv0SBVlSLftRL24dwdkfRzeyAjDYDM68fQ/PQLuHoEshV7YtSduqJesylpf/0A2W83mqSsFDmWKPu+vQX95vWwmj6YoKW7SH7kiZatNZVWTCQzIpbwzX+8Z2kV1ejXkg7f5B93To3NPe4UDan0407RkBWPs/7O+b1z0QQR6urNmJvfU2twG9x+OVs4M3WGtuPl37fJ/iBDXgsHzHvVYXlz6tuKoes+l7/eNT5vkr4iFatk3VvaP3sbIzZOYvX9HWRnZRP01I+HJ25RqZ7de233rSn9HpaeTb9He0ymjCZs5gqyY/Ib7OnuHqS7509slfboGZWObcNwZF+i1+8oq1K/ByW/a94gl3GftljOGsGrz9eQHR1feoaPgWiQlko0SD8C9vb2SCQSnj9/Tr9+/Yq87+HhgYmJSakTAhUe7luYmppakR8+mZlvdqKW5A1H9Pf3p2fPnkyePJnVq1dToUIFbt68yYQJE954W5D7A0yiZIhj4fWFG9cSiYT/Y+++w6I4+gCOf4929N6RomIXNKLGir33FnuJmlhjjIlGYy+JiYnGqFFTjL3FaDSWaOy9gWJHAaV3pHe4e/9ADw4OQb2TxHc+z7PPA3sze/O7OWaZndlZ2Qvm3y9dupSFCxcq7TPQs8RQWr5Vio8eOYWvT2FHUk9PDwBbO2tiYgrvQbK2sSwxavoicrkcvxt3qFLVTbFvweIZ/PD9z/y5t2BE9MH9Rzg7OzJ12rhX6pCG/nODuJtBit+19QqaBwMbMzJjkxT79a1NS4xeFJUZl4SBrfIIYkGeglGq7KepyPLyMXh2xVyRxspMMbJq924N9K1NGXDtB8XrWjraNJw3lNpjO/NHk08AiDx3l33NP0VqYYw8X0ZOSgYDb67hiRru95JnpCCX5ZcYDZUYmSJPK/8qvrLwQHQ8WxQeNzUJZHlK/7zI4yLQMrEAbW3Iz1dxFOFVyDPTCurQqFgdGpiUGDVV5ElPRp6WpOiMAsieRiORaCExsUCeVHhBS6dBB3QbdSZ730rk8S9eAVYT8p+mIM/LLzEaqmNtRm6xUdOX4fDZEJ7uO0PCroJbLbIehqBlKMXl60nErN6j1o7Sk+M3iPEr2e4Y2piRUaTdMbA2JSOu9HYnIy5JaTQUwNDalIwXrLidl5lNgn8Y5pXtSrzm2LgGFu6OHJ24pryhvJLcpynI8vLRK9Ye6lmblRg1/S+5e8KXEL/Clax19ArOxSa25qQUGVkzsTYj9QXnk/JICI1h9cBF6BlI0Tc2ICUuiZFrPiYhrPSLz+qUn1jwd6hjbUHRy9nalmZl3vNp1LkVNos+IebTL8m8UsaKs3I52Xcfoefi9PqFfg2KeFW0O8VnaxRn1r0Flb6ZQsjEr0m7eOuFaYX/L+Ie0reAlZUVHTp0YO3atWRmZiq9Fh0dzfbt2xk4cCAeHh7IZDLOnj2r8jienp6cP3++1I6hjY0NUVGFoz/5+fncvVtySfMrV66U+L1mzYJ7OXx8fMjLy2P58uU0adKE6tWrExmpPPVET0+P/DL+Ka9duzahoaGEhYUp9t2/f5/k5ORSpy6Xx6xZs0hOTlbaDPTKP/UtLS2dJ49DFdtD/0BiomNp3aZw0QZdXV2aNW/M9as3XqpsdT1rERNdeII1MNRHXqxznZ8vQ0vr1e5FzEvPIjU4RrElPYogIyYJR++6ijRautrYN6lJrE9AqceJ8w3EsWVdpX2O3h6KPLLcfBJuP1E6bkGauoo0QXsvcqD9F/zVcbZiS496yr11h/ln6LIS75mdmEZOSgb2zWujb21K2PGX+2xVys9HFvkE7aqeSru1q3qQH/ao3IfRcnAr6OA8P2zYQySW9kr3jEqsHJClJorOqLrJ8pHFhqLlotwmaLvUUlqkqKj8qKCC1XJ1pYp9WhZ2yGUy5KmF/1zqNOiAbuOuZO9fjSy2YhakkufmkXEnCJOW9ZT2m7SsT7qv6keGlIeWgRTkxS7c5csKRrPUfK9zbnoWycExiu3powjSY5Jwaanc7ji9W5Mo39LbnegbgTgXa3dcvD2IfkEeLT0dLKs5kR6TVOK12oNaEXP7MfEPNFu38tx8Um8/xrKVcjtj6e1Jso/qx0v9F2SnZxEfEqPYogPCSY5NpEaLwkX3tHW1qfpuLZ74lr89fZGczGxS4pIwMDWiprcnd45r/p5nAPLyyL4fgEFT5XUhDJs2IOuW6segQMHIqO2ST4n5/GsyzpVvFWe9mlXIi6/YFdnluXlk3g3EuMU7SvuNW9Qnw7f02xbMe3rj/N1UQj/+jtTTmpsG/28kl8vVur2NxAjpW2LNmjU0a9aMTp06sWTJEipXrsy9e/eYPn06Tk5OfPnll1haWjJy5EhGjx7NqlWrqFevHiEhIcTGxvLee+8xefJkVq9ezaBBg5g1axZmZmZcuXKFxo0bU6NGDdq2bcu0adM4fPgwVatW5fvvvy/xrFCAixcvsmzZMnr37s3x48fZs2cPhw8XjOJVrVqVvLw8Vq9eTY8ePbh48SLr169Xyu/m5kZaWhonT56kXr16GBoalhjdbd++PZ6engwdOpSVK1eSl5fHxIkTadWqFQ0bNnzlz1EqlSKVSpX2lXe6bmnWr93M1E/HExQUwuOgYD75bDyZmZns3XNIkebHn5YRFRnDkoXLAZg+czI+1/14HBSCiYkxH4wfTl2Pmsz4tHD09tjfp/nkswmEh0fh/yAAD8/aTJj8Pju2/vFa5S3q/q9H8fyoJylPYkh5Eo3nRz3Jy8zh8Z+XFGla/DCOjKhEbnxdMJXv/oZjdNk7h7oTuxN2zBfnTl44tqzDkT6LFXnu/fI3LX+YQPytx8T5BlJ9WBuMnKx4uPUkUNDBzE5UvqdLnpdPZlwSKUGFF0Xc3/MmOTCCrIRUbLyq0XjRMO79clQpzevIvXwYad9JyCIfkx/2CF2vdkjMrMnzOQGAbrtBSEwtyPmzYPqUTpMuyJPikMWGg7YOOp4t0Kn9Llm7VyiOmXf9OLqNO6HXeSS5146iZemAXsve5F49qpYya0pGRiah4YUXjyIiY/B/FISZqQkO9rYVWLIXy7txAr1O7yOLCUEW9Rgdj5ZITCzIu3MOAN1mvZEYm5PzzyYA8h9eR964K3odRpB75RASfSN0W/QteAbps+m6Ol4d0W3Sg5xjvyFLSQDDZ/ej5mYXbG9Q7K8HcP1+Khm3A0m/8RDrIZ3Qc7QmflvB98nh8+Ho2VsR8slKRR6D2gVTGbWNDNCxNMOgdmXkuXlkBRRc4Es+cR3bsb3IuPuEDL+HSN0ccPhsKMnHr7+RFR79Nhyl4eSeJAXHkPQkmoaTe5KblcOj/YXtTofvx5EWncjlb35/lucY/f6YQ4MJ3Xnyjy+VO3pRqUUd9vYtbHeazxnMkxM3SYtIwMDKlEZTeqFnbID/H+eV3l/X2AD3bo25sHiHxmMFCFt/iNprPiL11mOSfR7hOLw90krWRG4uGKGuMnswUntLHnz0oyKPcR1XALSN9NG1MsW4jiuy3Dwyni0AJ9HVxqh6JaCg4y21t8S4jiv56VlkBsdQEc7+9jcdJvUmPjiauCdRdJjUh9zMbHwPXFSkGbp8IskxTzm0bBdQ0Gm1r1YQh46uNmZ2ljjVdlV0eAFqenuCREJsUCQ2bvb0/GIosY+juLrnzBuLLWnLPuyWTif73iOybj3AtH9XdBxsSdld8L+P5dT30bG1JvaLb4FnndGvphP/9Tqyb/mjbVVw4VuenY0srWBxNIsJQ8m65U9uaITiHlJpjarEL/lRdSHeoLhf9+O8YhqZtwPIuOGP5ZDO6DrakLC9YOq7/YwR6NpZEfbp98CzzujyT4hc+AsZN/3ReTYjQJaVgyz1zS4GVyHElN0yiQ7pW6JatWr4+PiwYMECBg4cSEJCAvb29vTu3Zv58+djaWkJwLp16/jiiy+YOHEiCQkJuLi48MUXXwAFI62nTp1i+vTptGrVCm1tberXr0/z5gWje6NHj+bWrVuMGDECHR0dPvnkE9q0aVOiLJ9++im+vr4sXLgQExMTli9fTqdOnQCoX78+K1as4JtvvmHWrFl4e3uzdOlSRowofL5ds2bNGD9+vCKO+fPnKx798pxEImH//v189NFHeHt7o6WlRefOnUs8YubfYPXKXzAw0OfbFfMxMzfjhs8t+vceTVpa4SMXKlVyUJpKbGZmwoofFmNrZ0NKSip3bt+nR5eh3PQtvAdq1vTFzJzzMcuWz8faxoro6Fg2b9zFd1+r72R1d+0hdPT1aPLVKKRmhsTdDOKfId+Ql144ndHY0VqpsY3zCeDsxDU0mDGAd6b3JzUkhjMT1hBfZDpw8F9XkVqYUP+TPhjYmpP4MJwTw78lPaL805gBzKo64DXrPfTMjUkLj+P2qr+4/3PJe8FeVf69K+QYmqDbqi96xubIYsPI2v4N8uSCR/BITMzRMitckEOirYNux6FITCwhLwdZbDhZ278hP8BPkUae8pSsrUvR6zwcgwnfIE9JJPfq3+Re+Ett5daEu/4BjP7oc8Xvy1b/DECvLu35cs6nFVWsMuUH+JJrYIzuu92QGJoiT4gk+8Aa5KkFowwSI7OC+nouN5vsP39Ar/Ug9AfNQp6VVnCMS4X1o+PZComOLtJu45TeK/fKIXKvHuJNSjp4AR1zE+w/HoiurSVZj0IIGrmI3IiCaeu6thYlnkla8+hKxc+Gnu5Y9mlFdlgM95sX3PMevep3kMtxnD4UXXtL8hJSSD5xnahvt72RmG6sK2h3Wi8paHdi/II4MPQbcou2O07WSiMF0b4BHJ20hqbTB9Dks/4kh8RwbOIapenAxg6WdFozCQMLEzKfphB9I5Dfe80ntVi7U71nE5BIeHTgsuaDBWIPXEbXwgS3af2Q2lmQ5h/G7SFLyQovaGekthboOynXYeNT3yp+Nq1fFft+LckMjeVyo8kFeewtldK4TuqJ66SeJF68x82+yrelvCkn1/+Frr4e/RePxtDMiBC/QNYN/4rsIvVqUaxezewsmXHkG8Xv7cb1oN24HgRcuc+aQQXrNeibGNJjxmDM7S1JT07j1t/XOPzdLmR5b27GSfrRs8SbmWAxfig6NpbkBIQQNWEOeVEFs5q0rS3RcShc0dn0va5IdHWwmfsRNnM/UuxP2f8PcXMKLkxrmRhjs+BjdKwtkKVmkO0fSMSoz8i+W/Ej58mHLqBjbordx4PQsSlod4LfX6hod3RsLdF1KozXckhnJLo6OC2ZgNOSCYr9T/84SfhnK9908YV/IYn8bR37FSqEm5sbU6dOLfG80v8ya9NXX5L+v+A7k7f7OWADPnj7p8HqTX4zq0ZXlNztJadpv20eLlfPqP6/1QW5SdmJ/sPq5ORUdBE07oDB232X1xTjf8czPjUlPU2voougcZ7BByu6CCqljFHvM3FNNxwvO9F/jBghFQRBEARBEARB0AC5mLJbprf7cpcgCIIgCIIgCILwryVGSAW1Cg4OrugiCIIgCIIgCMK/gxghLZPokAqCIAiCIAiCIGiC5hcn/88TU3YFQRAEQRAEQRCECiFGSAVBEARBEARBEDRALGpUNtEhFQRBEARBEARB0ATRIS2TmLIrCIIgCIIgCILwFlq7di2VK1dGX18fLy8vzp8/X658Fy9eREdHh/r162u2gIgOqSAIgiAIgiAIgmbI1Ly9hN27dzN16lRmz57NzZs3admyJV26dCE0NPSF+ZKTkxkxYgTt2rV7uTd8RaJDKgiCIAiCIAiCoAFymVyt28tYsWIFY8aMYezYsdSqVYuVK1fi7OzMunXrXphv3LhxDBkyhKZNm75O6OUmOqSCIAiCIAiCIAj/AdnZ2aSkpCht2dnZJdLl5OTg6+tLx44dlfZ37NiRS5culXr8jRs3EhQUxPz589Ve9tKIDqkgCIIgCIIgCIImqHnK7tKlSzEzM1Pali5dWuJt4+Pjyc/Px87OTmm/nZ0d0dHRKosaEBDAzJkz2b59Ozo6b27tW7HKriCUwduiZkUXQaO08iq6BJr19HBsRRdB46wsllV0ETRKd+iMii6CxmV+Pb2ii6BRlpKKLoFmZf0fXN83ectjzMzQregiaJTkLf8b/DdT92NfZs2axbRp05T2SaXSUtNLilW+XC4vsQ8gPz+fIUOGsHDhQqpXr66ewpaT6JAKgiAIgiAIgiD8B0il0hd2QJ+ztrZGW1u7xGhobGxsiVFTgNTUVHx8fLh58yaTJ08GQCaTIZfL0dHR4Z9//qFt27bqCaIY0SEVBEEQBEEQBEHQhJdcGVdd9PT08PLy4vjx4/Tp00ex//jx4/Tq1atEelNTU+7cuaO0b+3atZw6dYo//viDypUra6ysokMqCIIgCIIgCIKgAfIK6pACTJs2jeHDh9OwYUOaNm3Kzz//TGhoKOPHjwcKpv9GRESwZcsWtLS0qFu3rlJ+W1tb9PX1S+xXN9EhFQRBEARBEARBeMsMHDiQhIQEFi1aRFRUFHXr1uXIkSO4uroCEBUVVeYzSd8E0SEVBEEQBEEQBEHQhAocIQWYOHEiEydOVPnapk2bXph3wYIFLFiwQP2FKkZ0SAVBEARBEARBEDSgIqfs/le83Wt4C4IgCIIgCIIgCP9aYoRUEARBEARBEARBE8QIaZlEh1QQBEEQBEEQBEEDxJTdsokpu4IgCIIgCIIgCEKFEB3St4xEImH//v2lvn7mzBkkEglJSUlvrEyCIAiCIAiC8P9ILlPv9jYSU3bVYNSoUWzevJlx48axfv16pdcmTpzIunXrGDlyZJlLK7+MBQsWsH//fvz8/NR2zPIaNWoUSUlJL+z4CsoGTh1MhyEdMTIzJuDmI36Zu56wgLBS07/buSn9JvXHwdUBbV0dop5E8tcv+zn75xlFmtqN69BrXB+qelTF0s6Krz/4kmv/XFVruWuMbE/t8V0xtDUn6VEE1+dvI/baw1LT2zWpScP5QzGv7kRGTBL31h3i0dZTSmlcujai/vT+mLjakhoSy81v9hB21Efxer1pfan3aV+lPJmxSex5Z7JSGrdeTTB0tESWk8/TO0+4+c0e4m8GqSny0hn374nJsPfQtrYi93EwiSvWkuN3R2VavXp1Mf/oA3RcXZDoS8mPjiFt3yHSdu7VeDnLS8ezFToNOiAxMkOeEEnOuT3IIgNLz6Ctg27jbmjXbIzE0BR5WhK51/8m//6lgpfrtECn1rtoWTkCIIsNJffSAWQxwW8gmlfn43eHjTv+4L5/IHEJT/lh6VzaeTer6GKVi/3ITjhO7IWerQUZj8J4Mm8jqVcfqEyra2uO2/xRGHtWQb+KA1EbjhA8b6NSGpv32lDth8kl8l52G4Q8O1cjMRRVbWR7ak/oisGzdsd33jbiXtDu2DapSYMFhe3Og7WHCCjS7phVd8Jzej8sPStj7GyDz7ytPPz1mNIxJNpaeH7aF7e+zdC3MSczNonHv5/j7soDIJerNT6XUR2oMqkHUltz0h6Gc3/uFhKv+pea3rJpLWotHI5xjUpkxyTyeM1BQrecULz+7r55WDWvXSJf7PEb+AxbBoC2kT7VZ76HfZdG6FmbkXI3mPtzNpHs91itsZWl7dR+NBzcFgMzI8L9Ajk4dyOxARGlpret5kS7aQNw9KiMRSUbDi/awuXfjpaa3ntiTzrOGMSl3/7myKKtmgihVJbDumL9QV90bC3JfhRK1JJfyLh+T2Va005NsRzaFf1aVZDo6ZIdEErsDztIO39DkUZazQXbT4ZiUNcdvUp2RC3+mYSNf72pcMpkOawrNh/2RcfWguxHoUQu/oWM6/dVpjXt1BTLYV0wKBJvzA87SDt38w2XumK8rZ1IdRIjpGri7OzMrl27yMzMVOzLyspi586duLi4VGDJhIrWZ3xfeoztxS/zfubzHp+SFJfI/O2L0DcyKDVPWlIqe9fsYWbfGXzSaQqn9pxk8ncfU9/7HUUaqaGU4AdP+GXezxopt1vPd2m4YBh3Vv3FoU5ziLn2kHbbpmPkaKUyvbGzDW23fkbMtYcc6jSHO6v/otGiEbh0baRIY+3ljve6yTzee4GDHb7g8d4LtFo/Get3qiodK9E/jN/rT1Jsf7WbpfR6yuMors3ZzMF2szjaZxFpYfG03/E5UksT9X8QRRh0aI35tImkbNxB9LBxZPvdweaHpWjb2apML8/MIu33/cSOm0r0e++T8tt2zCa8j1GfbhotZ3lpV/NC13sAudf/JmvHl+RHBiLtNRmJiUWpefS6fICWcw1yTmwla8sCco5uQJ4YXXjMStXJf+RD1t7vyfp9GfLUp0j7TEFiZK75gF5DZmYWNdyr8MU01c9q+7ey6tkMt0XvE/7DXm51/IyUqw+ovX02ek7WKtNr6emS+zSF8FV7Sb8XXOpx81LSue45Rml7E51R157v4rVwGHdX/cWRjnOIu/qQNtunY+ikut0xcrahzbbPiLv6kCMd53Bv9V94LR6Bc5F2R9tASlpoHH5f7SYzJknlcWpP6o77iHZcn72FQ61mcHPJTmpP6EaN0R3VGp9Dr6bUXjySwJV/cqH9TJ5e9afRzpnolxKfgYsNDXd8ztOr/lxoP5PAH/ZT+8tR2HdrrEhzY/RyTtQdp9jOeX+GLC+fqIOFFyg9vh+HtbcHfpN/5Hzr6cSfuU3jPXOQ2pf+t65uLcf3oNmYLhyat4l1PeeQGpfMqG1foGekX2oeXQMpT0Nj+eebXaTGJr7w+E6eVWg0uC1RD0LUXfQymXZrif2cD4j78XeCuk8h3ecerr8tQNfRRmV6w8Z1SbvgR8iYBQT1mkr6ldu4/DIX/dpVFGm0DKTkhEYTs2wzubFP31Qo5WLWrQUOc8cS++PvBHb7mPTr93DbWHq8Ro3rkHbBj+DRCwnsOZW0y7dxLRav8P9NdEjVpEGDBri4uLBv3z7Fvn379uHs7Mw777yjlDY7O5spU6Zga2uLvr4+LVq04Pr164rXn0+rPXnyJA0bNsTQ0JBmzZrx8GHBFeJNmzaxcOFCbt26hUQiQSKRKI2+xsfH06dPHwwNDalWrRp//aX6ilp6ejqmpqb88ccfSvsPHjyIkZERqamp5Yq9devWTJkyhRkzZmBpaYm9vX2Jh+gmJSXx4YcfYmdnh76+PnXr1uXQoUOK1/fu3UudOnWQSqW4ubmxfPlypfxubm4sWbKEESNGYGxsjKurKwcOHCAuLo5evXphbGyMh4cHPj4+SvkuXbqEt7c3BgYGODs7M2XKFNLT08sVl7p0H9OTvWt+5+rRy4Q+CmXVpyuR6kvx7uVdap57V+5y9dgVIgLDiQmN5vDGg4T4B1OrUeFV8JtnbrDzu+1cPXpZI+Wu9UEXAnedIXDnGZIDI/GZv430yASqj2inMn314W1Jj0jAZ/42kgMjCdx5hsDdZ6kzvqsiTe2xnYk6d5e7aw6SEhTF3TUHibpwn1pjOysdS54vIysuWbFlP1X+Lj7Zf5mo8/dIC40j+VEEPgu3o2dqiEVtzV78MRnSn/QDf5N+4Ah5waEkrVhLfkwsxv17qEyf+yiQjH9Ok/c4hPyoGDL+PkHWFR+k9T00Ws7y0mnQnrx7F8m/dxF5YjS55/YgT0tEx6OVyvRarrXRrlSN7ANrkIX5I09NQBYTjCyqcJQl59hv5N0+izw+HHliDDkntwEStJxrvKGoXk3Lpo2Y8uFIOrRuXtFFeSmO43oQu/MUsTtOkhkQQfC8jWRHJmA/spPK9NnhcQTP/Y24PWfJT80o/cByyI1LUtrehJofdiFo5xmCdpwhJTAS3/nbyHhBu1NtREG74zt/GymBkQTtOMPjXWepVaTdeXrrMTcX7yTkwBXyc1R3qm28qhF+zJfIk36kh8cTdvg6UWfvYFWvslrjqzy+G2E7ThO+/TTpAZE8mLuFrIgEXEd1UJneZUQHssITeDB3C+kBkYRvP034ztNUnthdkSY3KZ2cuGTFZt3Kg/zMbKIPXgFAS18X+26N8V+8g8Qr/mQExxDw3R9khsaW+r6a0Gx0Z87+eID7x64T+yicvZ+uQ9dAj3q9Sp+JEHH7MceW7uDOwcvk5eSVmk7PUMqAlZPYP/NXspLf7DkewHpMbxL3HCfx93/IDgonevEv5EbFYzm0q8r00Yt/If7nvWTeDiAnOJKY77aQExyJSbvCCw2ZtwOI+XojyYfOIS/le1tRrMf2JvH34yTuLog3avGvz+LtojJ91OJfif9p37N4o4j5bis5wVGYFon3rSaXqHd7C4kOqRq9//77bNxYOPXpt99+Y/To0SXSzZgxg71797J582Zu3LiBu7s7nTp14ulT5Stgs2fPZvny5fj4+KCjo6M41sCBA/n000+pU6cOUVFRREVFMXDgQEW+hQsX8t5773H79m26du3K0KFDSxwbwMjIiEGDBimVGWDjxo30798fE5PyjzZt3rwZIyMjrl69yrJly1i0aBHHjx8HQCaT0aVLFy5dusS2bdu4f/8+X3/9Ndra2gD4+vry3nvvMWjQIO7cucOCBQuYO3duiSnO33//Pc2bN+fmzZt069aN4cOHM2LECIYNG6b4HEeMGIH82fSqO3fu0KlTJ/r27cvt27fZvXs3Fy5cYPLkklPRNMXO2Q4LW0v8zvsp9uXl5HHv6j1qeNUq93E8mnviWMWJ+1dVT/9RNy1dbaw8KxN59q7S/qizd7FpWE1lHhuvakQVSx955g5WnpWR6Gg/S+NO5Dnl6a2RZ2+XOKZJZTv6+66mz+UVtFw7CWMX1Vddn5e12tA25CSnk3hPg1fGdXTQq1mdrKvKFz2yrvqi51mnXIfQre6O1LMO2Tdua6KEL0dLGy1bF2ShylM780MeoOWg+qq1dpV6yGJC0G3YEf0xX6M/YiG6LfqBtm7p76OjB9rayLNf0PkRXolEVwdjz6oknfVT2p909hYmDV/vAoC2kT5e19fj5fszNbfMwqiuejtmqmjpamPpWblEOxJ19i7WL9vu1Ctsd8oj9voj7FvUwaSKPQDmtV2waVyDiFO3XjKK0kl0tTH1rEz8GeW//7iztzFvWF1lHouG1Yg7Wyz96duY1atSanzOQ9oQtf8y+RnZBe+rrY2WjjayYiPc+Vk5WDSu+arhvBQLZ1tMbC0IPF8YS35OHsFXH+DipTr2l9Fj8fs8PH2ToIt3y06sZhJdHQzqupN2Xnn6adr5mxg2KOfnK5GgZWxAflKaBkqoXi+Mt7z/10gkaBkZkJ9cvoGP/zpxD2nZxD2kajR8+HBmzZpFcHAwEomEixcvsmvXLs6cOaNIk56ezrp169i0aRNduhRcSfrll184fvw4GzZsYPr06Yq0X375Ja1aFYxUzJw5k27dupGVlYWBgQHGxsbo6Ohgb29fohyjRo1i8ODBAHz11VesXr2aa9eu0blz5xJpx44dS7NmzYiMjMTR0ZH4+HgOHTqk6EyWl6enJ/PnzwegWrVqrFmzhpMnT9KhQwdOnDjBtWvXePDgAdWrF5x4qlQp/Id3xYoVtGvXjrlz5wJQvXp17t+/z7fffsuoUaMU6bp27cq4ceMAmDdvHuvWraNRo0YMGDAAgM8//5ymTZsSExODvb093377LUOGDGHq1KmKcq1atYpWrVqxbt069PVLnyakLua2BdOhkoqNLiTFJ2HjVHonC8DQxJBfrm5EV08XWb6Mn+eu59YFPw2VVJnU0gQtHW2y4pOV9mfGJ+Noa64yj4GtGZnF0mfFJ6Olq4O+pQmZsUno25iTFVcsTVwyBjZmit/jbgZy8eOfSHkchYGNGR5TetPlwHz+ajuT7MTCk7VT+/p4r52MjoEemTFJHB/8jdLr6qZlboZER5v8p8rTxvITEtG3snxhXodDu9C2MANtbVJ+2UL6gSMaK2d5SQyMkWhpI89IUdovz0xBYmSqMo+WqTVaju6Qn0vOofVgYIxem8FI9A3JOaH6fi3d5n2QpyWV6PgKr0/H0gSJjja5xf6mcuOS0LMxf+XjZgaGEzB1DRkPQtA2McRxbDfq/vUlt9p9StaTqNcsdelKa3ey4pIxKKXd0bcxK9mmPGt3pJYmZMUmleu97685iJ6JAT3OLUOeL0OircWtr/cQsl99M1D0LE3R0tEmu1h5c+KSkZYSn9TWnJxi6bPjCuLTszQhu1h8Zu9UxaSWC7c/+UmxLz89i8Trj3D/pC9pjyLIjkvCsU9zzBu4k/44mjfB+Fkbn1YslrS4FMwrqZ5eXl4ePZriUMeN9b3mvtZxXpW2hSkSHW3y4kueG3RsGpTrGNZj+6BloE/ykfOaKKJaFcabpLQ/Lz4J3XK2O9Yf9EbLUErS4QvqL6DwnyQ6pGpkbW1Nt27d2Lx5M3K5nG7dumFtrdzQBgUFkZubS/PmhdPCdHV1ady4MQ8eKP/D5unpqfjZwcEBgNjY2DLvSS2az8jICBMTE2JjY1Wmbdy4MXXq1GHLli3MnDmTrVu34uLigrd36dNJy3rP5+V9/p5+fn5UqlRJ0Rkt7sGDB/Tq1UtpX/PmzVm5ciX5+fmKkdSi72FnZweAh4dHiX2xsbHY29vj6+tLYGAg27dvV6SRy+XIZDKePHlCrVolr+RlZ2eTnZ2ttC9fno+2pHxX2r17t2LcV4X3oH35/qLn76yUTiIpsauEzLRMPu0yFX0jfTyb1+P9OaOJCY3m3pU3dwVYLldV7hcUvJTXih6nRBKJRGln5OnCK+hJ/uHE+QTS59JyqgxoyYOf/1a8FnPxAYc6zkZqaUy1IW3wXj+Zv7svICtBuYOldiXKT5mLnsR+OBUtAwP0PGphNukD8sIiyPjntMaK+FJKVkjp381nX9zso79BThYAuef2oNftQzi9C/KVR2B0vDqiU6MRWXtXQH7p0+2E11P87xSJpKzm5YXSbgSQdiNA8fvDa/7U++dbHEZ34cnc317jyOVUIh4VMb4gvUSiev+LuPZqglu/5lyctJbkh+FY1HHFa+EwMmKSeLJH3Z2EkvG9TLv6PD5Vn4nzkDakPggludgCb7cm/YjHynG0u70OWV4+KXeeELnvIqYemhn5rterOT2/GqP4fevoZc/KrJxOUlbdlsHMwZJu80awacRS8t7APc4vVCIMSbm+g2Y9vLH9eAgh4xaTn5BcZvp/DVXtTjmq0qyHN3YfDyH4wyX/rXhfg1z2dk6zVSfRIVWz0aNHK6aE/vjjjyVef97wSiSSEvuL79PVLZwG9/w1mazssfqi+Z7nfVG+sWPHsmbNGmbOnMnGjRt5//33S5Tldd7TwKD0xXtAdeyqTlCqPo8XfUYymYxx48YxZcqUEscqrVO/dOlSFi5cqLSvpml1apmXb/rbtePXeHTzUWGZ9Qr+xMxtLEgssiCDmZU5ScWuLhYnl8uJDikYjQi+/4RK7pXoO7H/G+mQZj9NRZaXj0Gxq536VmZkxqk+gWTGJpdMb22GLDdPMXKZFZeEga1ZsTSmZMaX3onMy8wm0T8M08p2JfanBseQGhxD/I0gel/4DvfBrbi75mA5o3w5sqRk5Hn5aFspLwKibWlRYtS0uPzIaPKB3KAnaFtaYPrhyArvkMoz05DL8pEYKdeHxMCkxKipIk96MvK0JEVnFED2NBqJRAuJiQXypMILXzoNOqDbqDPZ+1Yijy99FU3h1eU9TUWel49esdE1XWsz9d7zKZeTdisQ/SoO6jumCs/bHX0V7UjxUdDnsuKS0S8Wv9RKud0pj3fmDub+moOEHCi47zLJPxyjStbU+aiH2jqkOU9TkOXlIy0Wn561WYlR0+eyY5NK1K/es3Y1t1h8WgZ6OPRuRsCyPSWOkxESw9U+i9A2lKJjbEB2bBL1f/6YzFDVF6tf14MTvoT5Fa7WrfPsXGhia0Zake+mkbUp6fGv3ilx9KiCsY0ZEw5+qdinraONa+OavDuiIwuqj0AuU+8qycXlJ6Ygz8tHx6bYucHKvMQoYnGm3Vri9PUUQid/TfpF9U0P16TS4tWxMiszXrNuLaj0zRRCJ/134lWHt3WarTqJe0jVrHPnzuTk5JCTk0OnTiUXlXB3d0dPT48LFwqnKeTm5uLj46NyxK40enp65Ofnq6XMw4YNIzQ0lFWrVnHv3j1GjhypluM+5+npSXh4OI8ePVL5eu3atZU+DyhYjKh69eqK0dFX0aBBA+7du4e7u3uJTU9PT2WeWbNmkZycrLRVN3Mv93tmpWcSHRKl2MICwkiMfUq9FvUVaXR0dajzbh0e+r7kFEaJBF29F9yrp0ay3HwSbj/B0buu0n4H77rE+QSozBPnG4BDsfSOreqScPsJ8rz8Z2kCcWhZLI23R6nHBNDS08GsmlOpq2MWkqCtyc8nL48c/0fov+ultFu/sRc5t1/i3l6JBInum6nHF5LlI4sNRctFud3RdqmltEhRUflRQQWr5epKFfu0LOyQy2TIUws75ToNOqDbuCvZ+1cjiw3VSPEFkOfmkXY7CHPvekr7zb09SfUp/TEpr8KoTmVyYl584eV1yXLzeXr7SYl2xMG7LvEv0e44tKpLwq3Cdqc8dPT1SnRc5Pmyl744+yLy3HxSbj/BupXyombW3h4k+ag+Pyb6BGDtXSx9a0+Sbz0uEZ9Dz6Zo6ekQ8UfpHej8jGyyY5PQMTPCprUnMcd8XzGaF8tJz+JpSIxiiw2IIDU2kaotCmPR1tXG7d1ahPqqjr08gi7eZVXHGfzYdZZiC78VxO39F/mx6yyNd0ah4O8w824gxkXO8wDGLeqTcaP0x/mY9fCm0rdTCZv6HWmnfUpN929TGK/ygp3GLeqT8YL/a8x6eFPpu6mEffwdqf+heIU3Q4yQqpm2trZi6q2qzpSRkRETJkxg+vTpWFpa4uLiwrJly8jIyGDMmDEl0pfGzc2NJ0+eKKbDmpiYIJVKy86ogoWFBX379mX69Ol07NiRSpUqvdJxStOqVSu8vb3p168fK1aswN3dHX9/fyQSCZ07d+bTTz+lUaNGLF68mIEDB3L58mXWrFnD2rVrX+t9P//8c5o0acKkSZP44IMPMDIy4sGDBxw/fpzVq1erzCOVSkt8juWdrluaQxv+ot+k/kQFRxL1JJK+kweQnZXNuQPnFGmmrJhKQvRTti/bAkDfif0Juh1IdEgUOno6NGjTkNZ92/DznHWKPPqG+ti7FY5Y2Drb4Va7MmlJqcRHxr9WmQEe/PI3zX+YQMKtx8T5BlJtWBuMnKx4tPUkAO/MfA9DBwsuflxwr9Kjraeo8X4HGs4fSsD209h4ueM+qDXnJxXOFHiw4Rid9s6hzsTuhB3zxbmTFw4t63C0z2JFGq+5gwk/fpP0iAT0rU3x+LgXusYGBD0bpdAxkOLxcS/C/vElMyYJqYUJNUa2x8jBguBD6n0Oa3GpO/7AauFMcu4/IvvOfYz7dEPb3pa0vQWjsmaTxqBtY83TBd8AYDygF3nRseQFF3TKpPXrYjJsAKm792u0nOWVd+MEep3eRxYTgizqMToeLZGYWJB3p+C7qdusNxJjc3L+2QRA/sPryBt3Ra/DCHKvHEKib4Rui74FzyB9Nl1Xx6sjuk16kHPsN2QpCWD47H7U3OyC7V8qIyOT0PBIxe8RkTH4PwrCzNQEB3vVj/X5N4j86SDVVk8h7VYQqb4PsRvWAamTNTFb/gHA5Yuh6NlbEjilsM0zrOMGFCxcpGtlimEdt4J/Mh+FA1Bp2gDSbgSQ+TgKbRMDHMZ0xbCOG49n/aLxePx//pumqyaQcPsx8T6BuA9rg6GTFQFbCtqd+rPew8DegsvP2p2ALQXtToP5Qwncfhrrhu5UHdyaixML2x0tXW3Mqjs9+1kHQwdLLOq4kJueTVpwDADhx29Sd0ov0iMSCqbs1nWj5rguBO06q9b4nqw/TL01k0i+9ZhEn0e4DG+PQSVrQjYXPFe0xuxBSO0tuf1RwfkvdMtxXMd0pNbC4YRuO4lFw+o4D2mD3/hVJY7tPKQNMUd9SoycQkEnFomE9KBIjNzsqTl/KOlBUYTvPKPW+F7k0m9HaTWpFwnB0SQ8iabVpF7kZuZw68AlRZp+yyeQEvOU48t2AwWdVptqlZ79rIOpnSX2tV0VHd6c9Cxin31vn8vNzCYjKa3Efk2K37CfSsunkXknkMwbD7AY3BldRxuebi9YL8Bu+kh07KyI+GwF8LxzNo2oxT+TedMfHWtzAGTZOcierX4t0dVB6u6s+FnHzgr9WpWRZWSRE6K5e7nLI/7X/VRaMY3MOwFk3PDH8nm8Owpuq7GbPgJdeyvCP/0eKIjXefknRC76hYxS4n2byd/SlXHVSXRINcDUVPWCIM99/fXXyGQyhg8fTmpqKg0bNuTYsWNYWJT/eWD9+vVj3759tGnThqSkJDZu3Ki0ANDLGjNmDDt27FC5KrA67N27l88++4zBgweTnp6Ou7s7X3/9NVAwkvn7778zb948Fi9ejIODA4sWLXqteKBgZPbs2bPMnj2bli1bIpfLqVq1qtKKxG/Cn+v3oacv5cMl4zEyNSbA7xGLhs0nK73wmbXWjjbIilzJlRpK+WDJeKwcrMjJyiEiKJwfpq7g4qHCkeSqnu4s3v2V4vfR88YCcGrPSdZ89sNrlzv4r6tILUzw/KRPwQPqH4Zzcvi3pEckAGBgZ46RY+E90mlhcZwa/h0NFwyjxsj2ZMQkcn3eFkKPFD7SKM4ngHMT1/DOjAHUn96f1JAYzk1YQ3yR+50MHSxp+eMkpJYmZCekEHcjkL97zFe8r0wmw7SqA61//rggTWIaCbcec7TvEpIfaXZqaObxMySZmWI6djja1pbkBgUTP3UW+dEF0960ra3QLtp5kUgwnzQGbUd7yM8nLzyKpDW/kr7vUCnv8GblB/iSa2CM7rvdkBiaIk+IJPvAGuSpBatyS4zMkJgUWbApN5vsP39Ar/Ug9AfNQp6VVnCMS4WPltLxbIVERxdpt3FK75V75RC5V/8dcaty1z+A0R99rvh92eqC5/v26tKeL+d8WlHFKlPCX5fQtTCh0rQB6NlakPEwlAfDviI7PA4APVsLpMWeSVr/ROFjtYzruWPT15ussFhuNJ4AgI6ZEVW/HY+ujTn5qRmk3X3C3T5zSSsyBVNTQv66ip6FCR5F2p0zwwrbHX1bc4yKxJMeFsfpYd/htXAY1Ue1JzMmEd+5Wwgr0u4Y2FnQ9XhhW1l7QjdqT+hGzKUHnOhfMNXTZ84W6s3oT+Olo5BamZIZk0jg1lPc+f5PtcYXdeAyuhbGuE/rh9TOnDT/MK4P+Zqs8IKLiFJbCwyKxJcZGofPkG+otWgELu93JDsmkfuzNxF9+JrScY2qOGDZpCbXBnyJKjqmhtSYPRh9B0tyk9KIPnSNR0t3vdQo8us6v/4guvp69Fz8PvpmRoT7BbFp+FJy0gtvATB3skJeZH6jiZ0Fk48sVfzeclx3Wo7rzpMr99kwaMkbK3tZUg6fJ9rCBNuPBqFjY0n2oxBCRi8gN7Lg71DHxgK9Is/otBzcBYmuDo6LJuK4qHDdicQ/ThAxY2VBHltL3A8XXkiy+bAfNh/2I/3KHZ4MUX4295uWfPgC2ham2E4pjDd49EJyIwri1bW1VHomqeWQzkh0dXBaPAGnxRMU+xP/OEn49JVvuvhvnJiyWzaJ/HXuJhfeGtu3b+fjjz8mMjKy1Oms/6/6uvas6CJoVO+8F19A+a9r41ixV5LfBKthqh+J8bbQHTqjoougcdc9pped6D/sieTFawn811n8Hyzadcng7b7La5CWhhfEq2D/D6N0Hk80s4bE6wp/t61aj1fp6im1Hu/fQIyQ/p/LyMjgyZMnLF26lHHjxonOqCAIgiAIgiCoiVhlt2xv9+UuoUzLli2jfv362NnZMWtWxU4BEQRBEARBEIS3iVyu3u1tJDqk/+cWLFhAbm4uJ0+exNjYuKKLIwiCIAiCIAjC/xExZVcQBEEQBEEQBEEDxJTdsokOqSAIgiAIgiAIggaIDmnZxJRdQRAEQRAEQRAEoUKIEVJBEARBEARBEAQNeFsXIlIn0SEVBEEQBEEQBEHQADFlt2xiyq4gCIIgCIIgCIJQIcQIqSAIgiAIgiAIggbI5WKEtCyiQyoIgiAIgiAIgqABcllFl+DfT0zZFQRBEARBEARBECqEGCEVhDL0zTWt6CJoVPRb3grcC7Wp6CJonN3yqIougkZlfj29oougcY3ufFvRRdCox57zKroIGiXn7Z+SJ33Lpx0mZuhXdBE0KlcuxqAqiuwt/9tRh7f8X1FBEARBEARBEISKIe4hLZu4XCIIgiAIgiAIgiBUCDFCKgiCIAiCIAiCoAHiOaRlEx1SQRAEQRAEQRAEDZDLK7oE/35iyq4gCIIgCIIgCIJQIUSHVBAEQRAEQRAEQQPkMolat5e1du1aKleujL6+Pl5eXpw/f77UtPv27aNDhw7Y2NhgampK06ZNOXbs2OuEXy6iQyoIgiAIgiAIgqABMrlErdvL2L17N1OnTmX27NncvHmTli1b0qVLF0JDQ1WmP3fuHB06dODIkSP4+vrSpk0bevTowc2bN9XxUZRKdEgFQRAEQRAEQRDeMitWrGDMmDGMHTuWWrVqsXLlSpydnVm3bp3K9CtXrmTGjBk0atSIatWq8dVXX1GtWjUOHjyo0XKKRY0EQRAEQRAEQRA0QN3PIc3OziY7O1tpn1QqRSqVKu3LycnB19eXmTNnKu3v2LEjly5dKtd7yWQyUlNTsbS0fL1Cl0GMkAqCIAiCIAiCIGiAXK7ebenSpZiZmSltS5cuLfG+8fHx5OfnY2dnp7Tfzs6O6OjocpV9+fLlpKen895776nlsyiNGCEVBEEQBEEQBEH4D5g1axbTpk1T2ld8dLQoiUR5hFYul5fYp8rOnTtZsGABBw4cwNbW9tUKW05vpEMqkUj4888/6d2795t4u1KNGjWKpKQk9u/f/0bf183NjalTpzJ16tQ3+r6l2bRpE1OnTiUpKQmABQsWsH//fvz8/BRpFixYwLp164iNjeXPP/9k//79b+Sz+7d8V8qj+sj21J7QFQNbc5IeReAzbxtx1x6Wmt62SU28FgzFvLoTGTFJ3F97iICtp5TSOHdtRL0Z/TFxtSU1JJZbX+8h7KiP4nUdI33qzeiPc5eG6FuZkngvGJ+520i49RgAiY429T/vj2Pb+pi42pCTkkn0+bvc/Go3mTFJaou92Sd98RzSBqmZEdE3gzgxdxMJjyJemKdal0a0+Kw/Zi62JIfGcn7ZHgKPFcYm0dai2Sd9qd27GYa25qTHJnFvzzkurzqgeIhXs0/6UqNHE0wdLcnPzSfmzhPOL9tDtF+Q2mJzHtWBypN6ILU1J+1hOP5zt5B41b/U9BZNa1Fz4XCMa1QiOyaRJ2sOErblhFIa1w+74DKyA/pO1uQ8TSXm0FUefbkTWXauIo3U3oIac4dg3bY+2vp6pD+O4u4nP5Fy+4naYlPFengXbMf1QdfWgqyAUMIXbiD92n2VaXVsLXCa8z6GHu5IKzsQt/EQEQs3lEhnM6YH1sO6oOdkTd7TVJKOXCLymy3Ii8T7JtmP7ITjxF7o2VqQ8SiMJ/M2knr1gcq0urbmuM0fhbFnFfSrOBC14QjB8zYqpbF5rw3VfphcIu9lt0EVFmN5+PjdYeOOP7jvH0hcwlN+WDqXdt7NKrpYJai7bTWr7kS96f2w9KyMsbMNPvO24v+r8qqRdSb3wKVrI0zdHcjPyiHOJ4CbX+4mJShK7fG5jupAlUndFW3MvblbSLxaenyWTWtRe+EwRRsTtOYQoUXamCb75mLVvHaJfLHHb3J92DIAXEa2x3VUBwycrQFIexhOwPJ9xJ26peboXsx7al8aDGmLvpkRETcDOTp3E3EBpZ87bKo50erT/jjUrYy5sw3HFm7l2m9HldK4NK5J03HdcPCojImdBb9/sIKH//hqOpQSHEZ1otLEnujZWpD+MIzH8zaR8oJ2psqCkRh7VsGgigORvx7h8bxNSmmsur6L88d9MXCzR6KrTebjKCLWHyT2j3NvIJqSnEZ1xGVST/RszUl/GE7A3E0kl3Ju1LM1x33hCEw8q2BYxZ7wX/8mYO5mpTSOw9phP8Abo5rOAKTefkzQVztJvam+8/m/xcsuRFQWVdNzVbG2tkZbW7vEaGhsbGyJUdPidu/ezZgxY9izZw/t27d/rfKWx0tN2R01ahQSiaTE1rlzZ02V75UEBwcjkUiUOlgAP/zwA5s2baqQMpXl6dOnTJ06FTc3N/T09HBwcOD9998vdRUsdfrss884efKk4vcHDx6wcOFCfvrpJ6KioujSpYvaP7sFCxZQv379Evufv9+/nWvPd/FaOIy7q/7icMc5xF59SNvt0zF0slKZ3sjZhrbbPiP26kMOd5zDvdV/0XDxCJy7NlKksfZyp+X6yTz54wKHO3zBkz8u0PKnyVi9U1WRpsnysTh41+XSR+s41G4WUWfv0m73TAzsLQDQMdDD0sONOyv3c6TTXM6NXYlJFQdab5pWokyvqvGE7niN7cLJuZvZ3n0e6XFJDNg+E10j/VLzODRwp8ePk7m37wJbOn/BvX0X6LF2Mvb1qyodt96wdpyct4WNbWdw7qudNBrXjQbvd1Skefo4ipPzNrOp4yx29ltEclg8A7Z9joGliVpis+/VlFqLR/J45Z9caj+TxKv+eO2ciX4p9WrgYoPXjs9JvOrPpfYzefzDfmp9OQq7bo0LY+/XnOqzBxO4fC8XWn7K3U9+wr5XE6rPHqxIo2NmRJODi5Dl5uM75GsueH+G/4Jt5CZnqCWu0pj3aIHT/DHErNmDf9dPSLt2n6qb56HraK0yvZaeLnlPU4hZs4fM+8Eq01j0boXj5yOIXrmLB20nEzp9NRY9WuD4+QgNRlI6q57NcFv0PuE/7OVWx89IufqA2ttno+dUeoy5T1MIX7WX9HvBpR43LyWd655jlLZ/c2cUIDMzixruVfhi2sSKLkqpNNG26hhISQuNe+GFObumtXi46ThHuy/gxKBvkGhr03bn52gblP1P38tw6NWE2otHELhyPxfaz+Lp1Yc0LqONabRjBk+vPuRC+1kE/nCAOl+OxL5IG+M7egUn6o5XbGe9pyPLyyfq4BVFmqyop/gv2cnFjrO52HE2CRfu0XDzZxjXqKTW+F6k2fjuNBnblaPzNrGhx1zS45IZun0Wei84d+gYSEkMjeXUN7tIjU1UmUbXUErMg1COFuvQvUnWvZpRZdEoQlfu40aH6aRcfUDdHV8gLa2dkeqSm5BC2A/7SL8XojJNXlIaYSv34tf9C260+ZSYXaepvnIS5q3raTIUlWx7NaXa4lEEr9zH9fafk3z1AfV2foG0lO/t8/hCVu4jrZT4zJvVJubPi9zsuxDfbnPIikig/u456D37f+ZtIpdL1LqVl56eHl5eXhw/flxp//Hjx2nWrPSLkTt37mTUqFHs2LGDbt26vXLcL+Ol7yHt3LkzUVFRStvOnTs1UTa1MzMzw9zcvKKLUcLTp09p0qQJJ06cYO3atQQGBrJ7926CgoJo1KgRjx8/1uj7GxsbY2VV2KgEBRVcnerVqxf29vZIpdI39tk9f79/u1ofdiFo5xkCd5whJTAS3/nbyIhMoPqIdirTVx/RlvSIBHznbyMlMJLAHWcI2nWW2uO7KtLU/KAzUefucm/NQVICo7i35iDRF+5T64OCCz7a+rq4dG3EzSW7iL36kLTgGG4v30daWJzifXNTMzk56BtCD14lJSiK+BtB+MzZglW9KqX+Q/eyGozpzNU1Bwg46kP8o3D+nvYTOvp61OpdeuPmNaYzIefvcu3HgzwNiuLajwcJvXgfrzGFF7McvaoR9I8vj0/5kRIez6Mj1wk+dwc7z8qKNP4HLhN64R7JoXEkPIrgzOLtSE0NsanlopbY3MZ3I3zHacK3nyY9IBL/uVvIikjAZVQHlemdR3QgKzwB/7lbSA+IJHz7acJ3nqbyxO6KNOZe1Um6/oiofRfJDIsj4extov68hGm9Koo0VT7qSWZkAnenrif5ZhCZYXE8PX+XzJAYtcRVGtuxvUjYfYKEXcfJDgwnYuEGciPjsR6u+qJQTngsEQt+5ene0+SnpqtMY9SgBum+D0g8cI6c8FhSz/uReOAchp5VVabXNMdxPYjdeYrYHSfJDIggeN5GsiMTsB/ZSWX67PA4guf+Rtyes+SnvuCCgBxy45KUtn+7lk0bMeXDkXRo3byii1IqTbStCbcec2PxTkIOXCE/R/VFg1NDl/H49/MkP4og6X4olz/5GeNK1lh5uqk1vsrjuxG24zRh20+TFhDJ/WdtjGspbYzriPZkhSdwf+4W0gIiCdt+mrCdZ6gysfCfxNykdLLjkhWbdSsP8jOziTp4VZEm9p8bxJ30I/1xNOmPo3m49Hfy0rOw8HJXa3wv0nhMZy6s2Y//UR/iHoVz4NP16OrrUbdX6eeOqNuPOfnVTu4dvEJ+dp7KNEFnbnHmuz34F5lN9KY5jetBzM5TxDxrZx7P20R2RAIOIzuqTJ8dFsfjuRuJ3XOWvFLameRL90j4+xqZARFkhcQQ+esR0u+HYNa4liZDUcl5fHcid5wiavspMgIiCJi7meyIeJxGqY4vKyyOgDmbiN5zjrwU1fHdn7iaiE3/kHYvhIzASPynrUeiJcGypYcmQ/m/M23aNH799Vd+++03Hjx4wCeffEJoaCjjx48HCqb/jhhReMF4586djBgxguXLl9OkSROio6OJjo4mOTlZo+V86Q6pVCrF3t5eabOwKLyaERAQgLe3N/r6+tSuXbtEr/zMmTNIJBLFdFEAPz8/JBIJwcHBin0XL16kVatWGBoaYmFhQadOnUhMLLg6dvToUVq0aIG5uTlWVlZ0795d0YkCqFy54B/Yd955B4lEQuvWrYGCEd6iU0Gzs7OZMmUKtra26Ovr06JFC65fv16irCdPnqRhw4YYGhrSrFkzHj4snFoTFBREr169sLOzw9jYmEaNGnHihPJ0vbLMnj2byMhITpw4QdeuXXFxccHb25tjx46hq6vLpEmTFGnd3NxYuXKlUv769euzYMECxe8rVqzAw8MDIyMjnJ2dmThxImlpaaW+f9HRygULFtCjRw8AtLS0FHPMi392MpmMb775Bnd3d6RSKS4uLnz55ZeK1z///HOqV6+OoaEhVapUYe7cueTmFvwjsGnTJhYuXMitW7cUo+zPR18lEonStOA7d+7Qtm1bDAwMsLKy4sMPP1SK5Xm5vvvuOxwcHLCysmLSpEmK94KCBwJXq1YNfX197Ozs6N+/f+mVUQ5autpYelYm6uxdpf1RZ+9i07CayjzWXtVKpj9zB6t6lZHoaANg4+VO1Nk7Smkiz9zG+tkxJdraaOlok19sFCY/MwfbxjVKLa+uqQFymUwto21mLjYY25oTfK6wnPk5eYRf9cfJS3XsAI4N3JXyAASfva2UJ+L6I1ya18Gisj0ANrVccGpUgyelTCnT0tXGc0gbspLTibuv+grsy5DoamPqWZn4M7eV9sefvY15w+oq85g3rEb82WLpT9/GtF4VRb0mXvPH1LMyZs9Gug1cbbFp9w5xJ24o8th29CLl1mPq/zKVNvd+otmJpVQa1va1Y3oRia4Ohh5VST3np7Q/5bwfRl41X/m4adcfYFC3Kob1CupWz8UO0zZepJx681PoJLo6GHtWJemsn9L+pLO3MGlY+t9MeWgb6eN1fT1evj9Tc8ssjOpWLjuT8EKaaltfha6pIQDZSaovvLwKia42Zp6ViSvWxsSdvY3FC9qYuGJtTNzpW5gVaWOKcx7Smqj9l8nPyFb5OloSHHo3RdtQSqJPwMsH8grMnW0wsbXg8Xnlc0fIVX8qveDc8V8g0dXBxLMKiWeUz1WJZ29h2uj12pmizFt4YODuSPIV1bdUaIpEVxsTzyo8LRbf07O3MXvNdrQobQMpEh0dcpNK/3/1v0rdixq9jIEDB7Jy5UoWLVpE/fr1OXfuHEeOHMHV1RUomJlYdDbmTz/9RF5eHpMmTcLBwUGxffzxx+r8SEpQ6z2kMpmMvn37Ym1tzZUrV0hJSXml+yb9/Pxo164do0ePZtWqVejo6HD69Gny8/MBSE9PZ9q0aXh4eJCens68efPo06cPfn5+aGlpce3aNRo3bsyJEyeoU6cOenp6Kt9nxowZ7N27l82bN+Pq6sqyZcvo1KkTgYGBSssbz549m+XLl2NjY8P48eMZPXo0Fy9eBCAtLY2uXbuyZMkS9PX12bx5Mz169ODhw4e4uJQ9aiOTydi1axdDhw7F3t5e6TUDAwMmTpzInDlzePr0abmXXNbS0mLVqlW4ubnx5MkTJk6cyIwZM1i7dm2ZeT/77DPc3Nx4//33iYoq/d6ZWbNm8csvv/D999/TokULoqKi8PcvvJfAxMSETZs24ejoyJ07d/jggw8wMTFhxowZDBw4kLt373L06FFF593MzKzEe2RkZNC5c2eaNGnC9evXiY2NZezYsUyePFlp+vDp06dxcHDg9OnTBAYGMnDgQOrXr88HH3yAj48PU6ZMYevWrTRr1oynT59y/vz5cn2OpZFamqClo01mvPLVosy4ZBxtzVXmMbAxIzKuWPr4ZLR0ddC3NCEzNgl9G3Oyih0zKz4ZA5uCzyYvPYs4n0d4TO1NckAEWXHJuPVuhnWDqqQ+UT2SpiXV5Z0vBhL852Vy0zJfMeJCRjYF8aUXK2d6fDKmpUxNep5PVR5Dm8J6v7b2IFITA0afXoYsX4aWthbnv92D/1+XlfJVaVef7msmo2ugR1psEn8M/YbMxNc/gelZmqKlo01OsXrKiUtGWkq9Sm3NVabX0tVBz9KE7NgkovdfRs/KlHf/WggS0NLVIXTjPzxZ/Zcij4GrLc4j2xP80xGCftiP+TtVqbVkFLLsXCL3vN73tTTalqZIdLTJi09S2p8Xl4SuzatPmUo6eB4dK1Oq7V1acMFJV4e4LUeIWbv3NUv88nQsTZDoaJNbrI5y45LQe/ZdfhWZgeEETF1DxoMQtE0McRzbjbp/fcmtdp+S9UT99xz+v9BU2/oqGi4YSuzVhyQ/DH+l/KqU1sZkxyUjtS15DoSCNia7jDamKLN3qmJay4Xbn/xc4lgmtZxpdngRWlJd8tOz8H1/BWll3PuvLsbP6i8truR5wOwF547/At1n7Uzxes2NS0b3NdoZAG0TQ971+wmJni7kywic9StJ526XnVGNdF9wbtQr5e/yVVSdM5Ts6KckFrt4/TZQ9z2kL2vixIlMnKj6Vo3it+OdOXNG8wVS4aU7pIcOHcLY2Fhp3+eff87cuXM5ceIEDx48IDg4mEqVCu5L+Oqrr176nsBly5bRsGFDpQ5UnTp1FD/369dPKf2GDRuwtbXl/v371K1bFxsbGwCsrKxKdPKeS09PZ926dWzatElRvl9++YXjx4+zYcMGpk+frkj75Zdf0qpVKwBmzpxJt27dyMrKQl9fn3r16lGvXuF8/iVLlvDnn3/y119/MXlyyUUviouLiyMpKYlatVRPwahVqxZyuZzAwEAaN26sMk1xRS8CVK5cmcWLFzNhwoRydUiNjY0VU3NL++xSU1P54YcfWLNmDSNHjgSgatWqtGjRQpFmzpw5ip/d3Nz49NNP2b17NzNmzMDAwABjY2N0dHRKfQ+A7du3k5mZyZYtWzAyMgJgzZo19OjRg2+++UZxQ7aFhQVr1qxBW1ubmjVr0q1bN06ePMkHH3xAaGgoRkZGdO/eHRMTE1xdXXnnnXdKfU9Vz3bKleejK1FxJbrYZSqJpGDlslKpSF+wu8j+EtklSvkufrSepis+oN/NNcjy8nl6J5jgPy9j4eFW4u0kOtq0XDcJiZYW12ZtKr1cL1CrdzM6LB2t+H3fqO9UllMikagoezEq8xTurNGjCbX6NOfQR2tJeBSObR1X2swfRnpMEvf+KOyUhV16wJbOszGwNMZzcBt6rJ3M9l4LyEhIeZUQy1HQl6tXitWrZbPaVJ3ah/szN5B0IxBDN3tqLRlJ1Zgkgr7fV5BFS4vkW48J+GoXAKl3gzGuWQmXUR001iEtLH6Jinn5S7BFGDepi/3kAYTP+Yn0m4+QujlQacFYcmMTiVn1+2uW9tWoivHVI4S0GwGk3SgcWXp4zZ96/3yLw+guPJn722scWQA007a+hEZfjcS8ljP/9F78SvnLUrxUxdvCcmR4tr9kHuchrUl5EEqyioVh0gIjOd92JrpmRth3b0y9VRO40meRRjqldXs3o9tXYxS/73z/W9UJJZJXrqd/HVXngtcMLT8tkxvtpqNtpI95Sw+qLBhJVkgMyZfuvd6BX4mq+NRTdy6TemLXpzk3+i5QWuxP+P/x0h3SNm3asG7dOqV9z0fuHjx4gIuLi6IzCtC0adOXLpSfnx8DBgwo9fWgoCDmzp3LlStXiI+PRyaTARAaGkrdunXL9R5BQUHk5ubSvHnhvTS6uro0btyYBw+UV0Xz9PRU/Ozg4AAUrFDl4uJCeno6Cxcu5NChQ0RGRpKXl0dmZqbaFiN63lCXNsqryunTp/nqq6+4f/8+KSkp5OXlkZWVRXp6uqJj9zoePHhAdnY27dqpvqcH4I8//mDlypUEBgaSlpZGXl4epqamL/0+9erVUypz8+bNkclkPHz4UNEhrVOnDtrahR1GBwcH7twpuMLWoUMHXF1dqVKlCp07d6Zz58706dMHQ0NDle+5dOlSFi5cqLSvj7EHfU0KvwPZT1OR5eVjUOzKp761GVnFriA+lxmXjEGxK4n6VmbIcvPIfja6lxWXhH6REcOCY5qSGV/Y0UoLieV4vy/RNpCiZ2JAZmwSLdZPJj00TimfREeblj99hLGzDcffW/rKo6OBx28QVeQfG21pQZNhZGNGepEr84ZWpmTEq44dID0uCaNisRXkKYyt1ezBXFt7kIfPFuKIfxiOqZM1jSf2UOqQ5mZmkxQSQ1JIDFE3gxhz9jvqDmrFtR8PvlKMz+U8TUGWl19i5EzP2qzEleHnsmOTSlwh1rMuqNfcZ/Xq/vl7RO45T/j20wCkPQhD21BK3e8+IGjlnyCXkx2TSNoj5ZGYtEeR2HV797ViepH8pynI8/JLjIbqWJuRW2zU9GU4fDaEp/vOkLCr4HaNrIchaBlKcfl6EjGr96jtH5jyyHuaijwvv0Qd6VqbqfeeT7mctFuB6FdxUN8x/w9pqm19GQ2XjKBSxwb802cJGVFPXzr/izxvY6TF2kI9a1Oy41RfUMuOTSoxeqpnbYosN4+cYvFpGejh2LsZj5btUXkseW4+GcEFs2mSbz3GvH4V3D7ozN3pJVfKfl2Pjt8gosi5Q0ev4NxhbGNGWpFzh5GVaYnZM/81uS9qZ16jLQVALicruGCF1PR7wRhWc8L5oz5vtEOa+wrnxpfhPKEHrh/3wW/AYtLva34hz4rwMgsR/b966XtIjYyMcHd3V9qed0hVXeUq/pwbLS2tEmmL3u8HBVNVX6RHjx4kJCTwyy+/cPXqVa5eLbhxPycnp9xxPH//8jybR1dXV/Hz89eed4KnT5/O3r17+fLLLzl//jx+fn54eHiUuyw2NjaYm5tz/77qewL8/f3R0dFR3BerpaVV4nMu+vmFhITQtWtX6taty969e/H19eXHH38ske51lFU/V65cYdCgQXTp0oVDhw5x8+ZNZs+e/VL1Ay9+TlLR/UXr5/lrz+vHxMSEGzdusHPnThwcHJg3bx716tVTuoe5qFmzZpGcnKy09TCuo5RGlpvP09tPsPdWvvhh712XuFLux4n3DSiR3qFVXRJuPUGeVzAVPc43EIcSaTyIV3HM/MxsMmOT0DMzxLGVB2HHCu/Pe94ZNa1sx4mBX5f4p+Vl5KZnKTp/SSExJDyKIC02CdeWheXU0tWm0rs1ifAt/V6kyBuBSnkA3Lw9lPLoGughlyl/t2UyGRKtMhpyiQQdPd0XpykHeW4+KbefYN1KeUEFa28PknweqcyT5BOAtXex9K09Sbn1WFGv2iriQiYrGOV4Flri9UcYVXVUSmJU1YHM8PjXiOjF5Ll5ZNwJwqSl8oqNJi3rk+5b+mNuyqJlIAW5THlnvqwg1nI890yd5Ll5pN0OwtxbOUZzb09SfUp/zMarMKpTmZwY1auACuWjqba1vBp9OQKXLg05MeAr0sPiys7wkuS5+STffoJNK0+l/dbeHiS+RBtj09qT5CJtzHOOPZugpadDxB8XylcgiQQtNbSdquSkZ5EYEqPY4gIiSI1NpHKLwli0dLVxfbcm4S84d/wXyHPzSL39GPNi9WrRypOU6+ptZ5BIkEg1U2elkefmk3r7MZbF4rP09iT5NdtRl4k9qDytH7cGf0XqLc0u4FmRZHKJWre30Ut3SF+kdu3ahIaGEhkZqdh3+bLy/V/Pp9MWvT+x+ONZPD09lR5DUlRCQgIPHjxgzpw5tGvXjlq1aikWO3ru+Wji83tOVXF3d0dPT48LFwob7tzcXHx8fEqdPqvK+fPnGTVqFH369MHDwwN7e3ulxZnKoqWlxXvvvceOHTtKPCcoMzOTtWvX0qdPH8U9ljY2NkqfXUpKCk+eFD6n0MfHh7y8PMXqWNWrV1eqD3WoVq0aBgYGpdbRxYsXcXV1Zfbs2TRs2JBq1aoREqK86Iyent4L6wcKvk9+fn6kpxcuKnHx4kW0tLSoXl31AhCq6Ojo0L59e5YtW8bt27cJDg7m1KlTKtNKpVJMTU2VNlXTdR/8/DfuQ1pTdZA3pu6OeC0YipGTFQFbCj6T+rPeo9kP4xTpH205hXElK7zmD8XU3ZGqg7ypOrg199cfUaTx//UYDq08qD2pO6buDtSe1B2HlnV48EvhM9ccWnng0NoTI2cb7L3r0v6P2aQERRG0u+C5ZBJtLbx/mYJVvcpcmLwOibYW+jZm6NuYoaX76gt8FHVjw1HendQT904Nsa5eiS7Lx5GXlcOD/ZcUabp8P46Wn79XmOe3Y7h5e9B4QncsqzrQeEJ3XFrUwXdDYWxBJ27S5KNeVGlbH9NK1rh3akjDsV0UzyrVNZDSYsZ7OLxTFVMnK2zrutHxm7GY2Fvw8HDhapKvI3j9YSoNbYvT4NYYVXOk5qIR6FeyJnRzwb3O1WcPwmN14X0YYVuOo+9sTc2FwzGq5ojT4NZUGtKGJ2sPKdLE/XMDl1Htse/dFAMXG6y8PXD//D1i//GFZx3V4J8OY+7lTpWPe2PoZodD3+ZUGt6W0I3Kz0pUt9hfD2A1qAOW77VD6l4Jp3lj0HO0Jn5bQb04fD4c1++nKuUxqF0Zg9qV0TYyQMfSDIPaldGv5qx4PfnEdayHdcG8R0v0nG0xaVkPh8+Gknz8ekFH/A2L/OkgtkPaYTuoLQbVnHBbOAqpkzUxW/4BwOWLobiv+kgpj2EdNwzruKFtpI+ulSmGddwwqF4486fStAGYt66P1MUOwzpuVF0xEcM6bopj/ltlZGTi/ygI/0cFI1cRkTH4PwoiKjq2gktWSBNtq5auNhZ1XLCo44KWrg6GDpZY1HHB2K3wOXyNvhpF5b7NuTBpLblpWYp2U1tfvf/8P1l/GOehbag0uDXG1RyptWg4BkXamBqzB1Fv9QRF+pAtJzBwtqbWwmEYV3Ok0uDWOA9pw+O1h0sc23lIG2KO+ihmZxRV44uBWLxbAwNna0xqOVNj1ntYNatN5N6Lao3vRa5tOEqLST2p0akhNtUr0Wv5eHKzcrh7oPDc0WvFeNrOGKj4XUtXG7vartjVdkVbTwcTewvsarti4VpYd7qGUkUaKFhAya62K6aO6llZvjwifjqI/ZB22A0uaGeqPGtnop61CW5fDKH6auV2xqiOG0aKdsYMozpuGBZtZz7qg7m3J/outhi4O+I0rju2A1pVyHNIw9YfwnFoOxwGt8GwmhPui0YirWRN5OaCmTBVZg+m1upJSnmM67hiXMdV0Y4a13HFsLqT4nWXST2pMnMQD6auIys0Fj0bM/RszNA2/Pc/aUFQv5eespudnV2i46Sjo4O1tTXt27enRo0aiuWCU1JSmD17tlJad3d3nJ2dWbBgAUuWLCEgIIDly5crpZk1axYeHh5MnDiR8ePHo6enx+nTpxkwYACWlpZYWVnx888/4+DgQGhoKDNnzlTKb2tri4GBAUePHqVSpUro6+uXWDTHyMiICRMmMH36dCwtLXFxcWHZsmVkZGQwZswYysvd3Z19+/bRo0cPJBIJc+fOVYzOldeXX37JyZMn6dChA8uWLaNu3bo8efKEOXPmoKWlxQ8//KBI27ZtWzZt2kSPHj2wsLBg7ty5StNVq1atSl5eHqtXr6ZHjx5cvHiR9evXv1R5yqKvr8/nn3/OjBkz0NPTo3nz5sTFxXHv3j3GjBmDu7s7oaGh7Nq1i0aNGnH48GH+/PNPpWM8X3DJz8+PSpUqYWJiUuJxL0OHDmX+/PmMHDmSBQsWEBcXx0cffcTw4cPLfKDvc4cOHeLx48d4e3tjYWHBkSNHkMlk1KjxeivDhfx1FamFCR6f9Cl4ePvDcE4P+5b0iAQADGzNMSqyUEN6WBynhn1Hw4XDqD6qPZkxifjM3ULYkcJVneN9ArgwYQ31Ph9Aven9SQuJ4fz4NSQUmfaka2rIO7Pew9DBkpykdEKPXMPv6z2KK+WGDpY4d/ICoPuJr5TKfLzfl8RcVv2Q7pdxbd0hdPT1aP/lKPRNDYnyC+KPod+Qm56lSGPqaK00KhjpG8ChyWto/tkAmn/an6SQGA5NWkO0X2FsJ+dtocVn/Wm/ZBQG1qakxyRya/spLv9Q8N2RyWRYVnWgTv+PMbAwISspjehbj9nVfwkJaroHKvrAZXQtjHGf1g+pnTmp/mH4DvmarGcjlVJbCwyK1GtmaBy+Q76h5qIRuLzfkayYRB7M3kTM4WuKNEHf70Mul1Nt5kD07S3JSUgh9h9fApbuVqRJ8XvMzfdXUH32IKpO60tmaBz+c7cQpeF/FpMOXkDH3AT7jweia2tJ1qMQgkYuIjeiYHRI19aixDNJax5dqfjZ0NMdyz6tyA6L4X7zDwGIXvU7yOU4Th+Krr0leQkpJJ+4TtS32zQaS2kS/rqEroUJlaYNQM/WgoyHoTwY9hXZ4QUx6tlalHhWYP0Theck43ru2PT1JisslhuNCzoKOmZGVP12PLo25uSnZpB29wl3+8wlzS/wzQX2Cu76BzD6o88Vvy9bXbDwTa8u7flyzqcVVSwlmmhbDews6Ha8sD2sPaEbtSd0I+bSA473L1gdvsaogge/d9xXuP4BwKWpP/H4d/Xdxx114Ap6FiZUm9YXqZ05af5hXB/yjWI2hNTWvEQbc33IMmovGo7r+x3Jjknk3uzNRBdpYwCMqthj2aQmVwcot/vPSW3MqL9mElI7c/JSM0i9H8q1QV8T/wYXkLm0vuDc0WXJKAxMjYjwC2L7sK/JUTp3WCmdO0zsLPjw78KYmo3rTrNx3Qm+fJ+tgwrqztGzCiN2F9Zbx3nDAbi15xx/ffaTpsMCIP5AQTvjMq0/erYWpPuHcnfoV2Q/q1c9u5LtTIOT3yl+NqlXFdt+LckKi+V6o4KLntqGUty//gA9B0tkWTlkBkbycPIq4ot04N+U2AOX0bUwwW1aP6R2FqT5h3F7yFKlc6N+sfganyq8b9i0flXs+7UkMzSWy40K1ldxGtURLakuHr8ptz1Pvt3Dk+9UTzv/r3pL7pLWKIn8Je4mHzVqFJs3by6xv0aNGooVVh89esSYMWO4du0abm5urFq1is6dO/Pnn38qHhty8eJFJkyYQEBAAI0aNWLKlCkMGDCAJ0+e4ObmBsDZs2f54osv8PX1xcDAgHfffZddu3Zhbm7OiRMnmDJlCo8fP6ZGjRqsWrWK1q1bK73Hr7/+yqJFi4iIiKBly5acOXOGUaNGkZSUpHisSFZWFjNmzGDnzp2kpqbSsGFDvv/+exo1Knig9pkzZ2jTpg2JiYmKhX78/Px45513FGUNDg5m9OjRXLlyBWtraz7//HP27NlD/fr1FY9ncXNzY+rUqS9ccTg+Pp5Fixaxf/9+IiMjyc/Pp1mzZhw8eFBpdd2UlBQ++OADjh49ipmZGYsXL+b777+nd+/eike/fP/993z77bckJSXh7e3N0KFDGTFihCKOTZs2MXXqVMW01QULFrB//37FSPX+/fvp06eP0tTg4p+dTCZj6dKl/PLLL0RGRuLg4MD48eOZNWsWULCC8W+//UZ2djbdunWjSZMmLFiwQPGe2dnZDB06lJMnT5KUlMTGjRsZNWoUEolEqR7v3LnDxx9/zOXLlzE0NKRfv36sWLFCsbBW8XJBwaJOfn5+nDlzhgsXLjBnzhxu375NVlYW1apVY/bs2bz3XuHoXVm2OQ4rd9r/omi1rrX971O3lGfXvU3spK+/gvK/WWbuW/4lBRrdKWXRl7fEbs95FV0EjbIoY8bP2+CGvlon1f3rtM4u5TE5b4lc+dtdfwBtYypm4byyXHLoV3ail9As6s2vWq9pL9UhFd6cDRs2MHHiRHbv3q30/E/hzRMd0v820SH97xMd0v8+0SH97xMd0v820SGtOKJDWra3/yz/HzVmzBgsLS158OABnTp1KnMhIUEQBEEQBEEQ/l3EKrtlEx3Sf7E+ffpUdBEEQRAEQRAEQXhFb345v/+et3/8XhAEQRAEQRAEQfhXEiOkgiAIgiAIgiAIGiBHTNkti+iQCoIgCIIgCIIgaIBMLB9bJjFlVxAEQRAEQRAEQagQYoRUEARBEARBEARBA2Riym6ZRIdUEARBEARBEARBA8Q9pGUTU3YFQRAEQRAEQRCECiFGSAVBEARBEARBEDRAPIe0bKJDKgiCIAiCIAiCoAFiym7ZRIdUEMpQUyutoougUdEYV3QRNOqQwdu/3nqNfJOKLoJGWf4fnMsfe86r6CJo1MDbiyq6CBq1/p23u/4A6mfnV3QRNOqOrn5FF0GjDMQwnfAvJjqkgiAIgiAIgiAIGiCuBZRNdEgFQRAEQRAEQRA0QHRIyyZW2RUEQRAEQRAEQRAqhBghFQRBEARBEARB0ACxqFHZRIdUEARBEARBEARBA2SiP1omMWVXEARBEARBEARBqBBihFQQBEEQBEEQBEEDZGLKbplEh1QQBEEQBEEQBEED3v6nob8+MWVXEARBEARBEARBqBBihFQQBEEQBEEQBEEDxHNIyyZGSIXX5ubmxsqVKxW/SyQS9u/fr5H32rRpE+bm5ho5tiAIgiAIgiCok0wiUev2NhIjpG+ZUaNGsXnz5hL7AwICcHd318h7Xr9+HSMjI40cu7iBAwfStWvXN/Je6mAzogv243uja2tB5qMwwhZsIO3afZVpdW0tqDTvfYw8qiKt7EDsb4cJW7BBKU2NPUswaVq3RN6kkz4EjlyikRhUafZJXzyHtEFqZkT0zSBOzN1EwqOIF+ap1qURLT7rj5mLLcmhsZxftofAYz6K1yXaWjT7pC+1ezfD0Nac9Ngk7u05x+VVB0AuR0tHmxbT+1O5TX3MXWzITs0k5MJdzn29m/SYJI3G23Vqf5oPboeBmTEhfgHsnvsb0QHhpaZvNqgtjft641jDGYDQO084+O1OQm4FKdJUbVyL9h/2wMWjMmZ2lvz84bfc/sentEOqVeNP+lJnaBv0n9Xf2TmbeFpG/VXt0ogmn/XHzNWW5JBYLn+7h8dHC8vb+JO+vDutr1Ke9NgkfvOarPj9o7BtKo99YclObv50+DUiKlRtZHtqT+iKga05SY8i8J23jbhrD0tNb9ukJg0WDMW8uhMZMUk8WHuIgK2nFK+bVXfCc3o/LD0rY+xsg8+8rTz89ZjSMSTaWnh+2he3vs3QtzEnMzaJx7+f4+7Kgu+uulUvFqNPOWL0KhLjfRUx1isWo3+xGOtM7oFL10aYujuQn5VDnE8AN7/cTUpQlNrje1U+fnfYuOMP7vsHEpfwlB+WzqWdd7OKLlapKurvsKg2S0dTd1hbzi3Yyq0Nx1SmUQfXUR1wn9gdqa05qQ/DuTdvC0+vlv6dtWpai9oLhmFSoxJZMYkE/XiIkC0nFK833TcX62a1S+SLOXGTa8OWaSSG4hp90pfaQwvOgzE3gzg3ZxOJZdRflS6NaFyk/q5+u4cnR1W3+w0m9aDJzIHc+vUoFxeqbjtbLR1NnWFtubBgK7c1WH/P1Z/WlxpD26BnZkTczSCuzN5EUhkxu3ZtRIPp/TFxtSU1JBbfb/YQWiRmu3drUHdCN6w9KmNob8HJ0d8TesxX06EI/zJihPQt1LlzZ6KiopS2ypUra+z9bGxsMDQ01NjxizIwMMDW1vaNvNfrsujRHOcFo4lavYf7naeRdu0+1bbORc/RWmV6iZ4ueQnJRK3aQ+b9YJVpAj/4Gr93Rim2u20/Qp6XT+KhSxqMRFnjCd3xGtuFk3M3s737PNLjkhiwfSa6Rvql5nFo4E6PHydzb98FtnT+gnv7LtBj7WTs61dVOm69Ye04OW8LG9vO4NxXO2k0rhsN3u8IgI6BHrZ13biyaj9bus7lwIcrsajsQJ8N0zQab/vxPWkzphu/z9vItz2/ICUumY+2zUb6gnirNamD71+X+GHwIpb3nUtiZDyTts7GzM5CkUZqKCXiQQi/z9uo0fIX12BCd975oAvn5mxmd/d5ZMQl0WvHi+vPvoE7nddOxn/fBXZ2+gL/fRfovHYydkXqDyDhYRgbGkxSbDs6zFJ6vehrGxpM4sSnPyOXyQj6+5paYnPt+S5eC4dxd9VfHOk4h7irD2mzfTqGTlYq0xs529Bm22fEXX3IkY5zuLf6L7wWj8C5ayNFGm0DKWmhcfh9tZvMUi581J7UHfcR7bg+ewuHWs3g5pKd1J7QjRqjO6olrtJiPNxxDrFXH9K2jBjbbvuM2KsPOfwsxobFYtR5FuPNF8Ro17QWDzcd52j3BZwY9A0SbW3a7vwcbQOp2mN8VZmZWdRwr8IX0yZWdFHKVJF/h89V6eSF3TtVSYt+qtbYinPs1YS6i0YQsHI/5zrM4unVh7y7YyYGpXxnDVxsaLx9Bk+vPuRch1kE/nCAuktG4tCtsSKNz+gV/OMxXrGdbjUdWV4+kQevaDSW596Z0J16H3Th/JzN/PGs/nqWUX92DdzpuHYyj/ZdYHenL3i07wId107Gtlj9AdjWq0LtIW2Ivx9S6vEqv6H6e85jYnfqfNiFK3M2c7DbPDLjkui0cyY6L4jZxsud1usmE7j3Agc6fEHg3gu0WT8Z63cKY9YxlJJ4P5Qrc0oOprwt5Gre3kaiQ/oWkkql2NvbK23a2tqsWLECDw8PjIyMcHZ2ZuLEiaSlpSnyPZ8Oe+jQIWrUqIGhoSH9+/cnPT2dzZs34+bmhoWFBR999BH5+fmKfMWn7BbVtm1bJk9WvjKbkJCAVCrl1KlTKvPcunWLNm3aYGJigqmpKV5eXvj4+CiVseh7SySSEttzERERDBw4EAsLC6ysrOjVqxfBwcEv+Ym+GrsPexG/6wTxO0+QFRhO2IIN5ETGYzOis8r0OeGxhM3fQMLeM+SnZqhMk5+URl5ckmIzbVkfWWY2iYcuajIUJQ3GdObqmgMEHPUh/lE4f0/7CR19PWr1Ln0kwmtMZ0LO3+Xajwd5GhTFtR8PEnrxPl5jCj8LR69qBP3jy+NTfqSEx/PoyHWCz93BzrPgYkpOaiZ/DP2Gh4eukvg4iqibQZyatwV7zyqYOKr+x0Yd2ozuyrEf/+TWsWtEPQpj66c/omsgpWGvFqXm2Tx1Nee3/UPE/RBigiLZMfMnJBIJNZp7KNLcP+PHoeW7uXVMPZ2x8qo/pjPXVx8g6KgPTx+Gc/yTn9DV16P6C+qv/tjOhJ2/i++PB0kMisL3x4OEX7xP/bHK32VZnoyMuGTFlvU0Ven1oq9lxCVTpWMDwi89ICU0Ti2x1fywC0E7zxC04wwpgZH4zt9GRmQC1Ue0U5m+2oi2pEck4Dt/GymBkQTtOMPjXWepNb5wFsbTW4+5uXgnIQeukJ+Tq/I4Nl7VCD/mS+RJP9LD4wk7fJ2os3ewqqf+C4G1nsUYWM4YqxeLMXDHGYJ2naV2kRgTbj3mRhkxnhq6jMe/nyf5UQRJ90O5/MnPGFeyxsrTTe0xvqqWTRsx5cORdGjdvKKLUqaK/DsEMLK3oNXikfwzZS2y3PwSr6tTlXHdCN15mtAdp0kLiOTevC1kRiTgOrKDyvRuI9qTGZ7AvXlbSAuIJHTHaUJ3nqHKhG6KNLlJ6WTHJSs2G28P8jOziTp4VaOxPOc5pjO+qw/w+Fn9nfyk4DxY7QX1V+9Z/d348SBJQVHc+PEgERfvU69Y/ekYSmm/agJnPt9AdrLq/wWM7C1ouXgkx99A/T1Xe2xnbq86QMjfPiQ9DOf81J/QNtCjap/SY64ztjOR5+5yZ81BkoOiuLPmIJEX7lOnSMwRp29zY9kfhPz9ZmYIVQSZmre3keiQ/h/R0tJi1apV3L17l82bN3Pq1ClmzJihlCYjI4NVq1axa9cujh49ypkzZ+jbty9HjhzhyJEjbN26lZ9//pk//vijXO85duxYduzYQXZ2tmLf9u3bcXR0pE2bNirzDB06lEqVKnH9+nV8fX2ZOXMmurq6KtNev35dMQocHh5OkyZNaNmypSKWNm3aYGxszLlz57hw4QLGxsZ07tyZnJyccpX/VUl0dTDyqErKOT+l/Snn/DBuWFNt72M9uD1P/7qALDO77MRqYOZig7GtOcHn7ij25efkEX7VHyevaqXmc2zgrpQHIPjsbaU8Edcf4dK8DhaV7QGwqeWCU6MaPDl1q9Tj6pkaIJfJyE5RfdJ+XVbOtpjZWuB//rZiX15OHoFX71PFq3q5j6NnIEVbV4eMpLSyE2uQqYsNRnbmhBapC1lOHhFX/XF4Qf3ZN3BXygMQevY29sXymFe2432f1Yy4uIJOP07C1MWm1GMaWJvi2rY+93efebVgitHS1cbSszJRZ+8q7Y86exfrhqpjs/GqViJ95JmCjqRER7vc7x17/RH2LepgUqXgu2te2wWbxjWIeMF391W8KEabUmK0VhFj1CvEWJyuacGsmOyk9Fc+xv+rCv87lEjosHI8N9YfLnOK8OuS6Gpj5lmZuDO3lfbHnb2NZSPVbaiFVzXizhZLf+YW5vWqlPqddRnSmsj9l8nP0Py58Hn9hRWrv8ir/iXqoii7Bu5KeUB1/XkvGUXIKT/CL9xTfSCJhHYrx+O3/nCZU4TVxdjFBkM7cyLOKsccc8Uf21LaHigYIY0oFnPE2dsvzCP8fxL3kL6FDh06hLGxseL3Ll26sGfPHqZOnarYV7lyZRYvXsyECRNYu3atYn9ubi7r1q2jatWC6RT9+/dn69atxMTEYGxsTO3atWnTpg2nT59m4MCBZZalX79+fPTRRxw4cID33nsPgI0bNzJq1CilkcyiQkNDmT59OjVrFnTcqlV7QWNnU3ii/fjjj4mKiuL69esA7Nq1Cy0tLX799VfFe23cuBFzc3POnDlDx47qn073nI6lCRIdbXLjkpT258Ylo2tjoTrTSzKqXw3Dmq6EfLZGLccr13vamAOQHp+stD89PhlTJ9VTkZ/nU5XH0MZM8fu1tQeRmhgw+vQyZPkytLS1OP/tHvz/uqzymNpSXbxnDuTB/svkpGW+YkQvZvos3tQ45bKnxiVjWan0zlZxvT4fQnL0U/wv3ik7sQYZPosns1hdZMQlY1Kp9PoztDEno3ie+GSMitRfzM1Ajk/9iaQnURham9FwSm/6/zmfHe1mkqWiI16rf0ty07MIUtNVcamlCVo62mQVK2dWXDIGtuYq8+jbmJFVrG6z4pPR0tVBamlCVmxSud77/pqD6JkY0OPcMuT5MiTaWtz6eg8h+1V/d1/V8xiL119mXDKOpcRoYGNGZLEYM5/FqG9pQmY5Yyyu4YKhxF59SPLD0u+lFlSr6L9Dr4ndkefLuPWb5u851LM0RUtHm+xi38HsuGSkRcpdlNTWXGV6LV0d9CxNyC72nTV/pyqmtVy4Ne1ntZa9NM/rr0RdlKP+SvztFjsPuvdsgo2HG390n1fqcRo8q7/bb6D+njN81r6oanuMXxCzgY15yTY2LhmDUur+bSV7O9chUivRIX0LtWnThnXr1il+f77g0OnTp/nqq6+4f/8+KSkp5OXlkZWVRXp6uiKNoaGhojMKYGdnh5ubm1IH187OjtjY2HKVRSqVMmzYMH777Tfee+89/Pz8uHXr1gtX4Z02bRpjx45l69attG/fngEDBiiVSZWff/6ZDRs2cPHiRUUn1dfXl8DAQExMTJTSZmVlERQUpOowZGdnK43mAuTI89GTvOJIQrHJ/hIJyNW0yIn1oPZk+IeQ7hegluOpUqt3MzosHa34fd+o7wp+KBGXpOwbG1TmKdxZo0cTavVpzqGP1pLwKBzbOq60mT+M9Jgk7v1xXimvlo423ddMQiLR4sScTS8ZVeka9mrB4K8+UPy+bvTXBUUvXmcSSbnrsf24nnj1bM4PgxaSl616OqSmVO/djDZfF9bfwWf1VzKcsuuvZLjKn0FIkRGQBMKJ8g1kxIXl1BzQEr9f/i5xvNoDW/Hwz0vkq/szKRFcGX9zxV5TXCd7ib9T115NcOvXnIuT1pL8MByLOq54LRxGRkwST/acL/sAL0tFmV8lxldtixp9NRLzWs7803vxK+X/f/Nv+ju08XCj3uhO7O4651VCeXVltP9lpS/ypS2R1GVwa1IehJJ0U/V5/XVV692M1kXq7/BrnAdfVH/GDpa0WDCcg0O/KbVdtPFww3N0J37XcP1V6dOMZt8Uxnx8hOqYC86FLz5Widdf4vz5tpBRsT3StWvX8u233xIVFUWdOnVYuXKlYjahKmfPnmXatGncu3cPR0dHZsyYwfjx4zVaRtEhfQsZGRmVWFE3JCSErl27Mn78eBYvXoylpSUXLlxgzJgx5OYWNnzFp8ZKJBKV+2Sy8s9iHzt2LPXr1yc8PJzffvuNdu3a4erqWmr6BQsWMGTIEA4fPszff//N/Pnz2bVrF3369FGZ/syZM3z00Ufs3LmTevXqKfbLZDK8vLzYvn17iTxFR1aLWrp0KQsXLlTa94FJDT40fblptnlPU5Hn5aNbbNRCx9qMvPiklzqWKlr6elj0bEHk8p2vfawXCTx+g6giJ3ltaUGTYWRjRnqRq9SGVqYlrhYXlR6XpHQVvzBPiuL3VrMHc23tQR4+W5Qi/mE4pk7WNJ7YQ6lDqqWjTY+1H2HmbMPvg5aqdXT0zgkfgot08HX0Cr77prbmpBQZ7TaxNiX1BfE+1+6D7nSc1Js1Q5cQ6R+qtnKW15PjN4jxK1J/egX1Z2hjRkaR+jOwNiUjrvR4MlTVn7Vy/RWXl5lNgn8Y5pXtSrzm2LgGFu6OHJ2ovtH97KepyPLy0X82evGcvnXJUdDnsuKS0S/2Nyq1MkOWm0d2YvmnV78zdzD31xwk5EDBdzfJPxyjStbU+aiHWjukz2M0eIkYM1WMEOu/QozPNVwygkodG/BPnyVkRL2ZxVT+6/5Nf4eOjWtgaG3KqCs/KNJo6WjTYu5Q6o/pzOZmn7xUbGXJeZqCLC8fqa1yufWsTckupdzZsUkl0kutTZHl5pFT7DurbaCHY+9mPFy2R63lLir4+A12q6n+DIvVn4G1KZnPPgcbz8oY2pgx4EjhhR4tHW0c362Bx6gO/FR1FA6Na2BgbcqIYvXXbO5QPMd0Zpua6i/0nxvE3SwZs4GNmdKsCgNr0xKzUorKjEvCoFhd6lubkvWC76ygXrt372bq1KmsXbuW5s2b89NPP9GlSxfu37+Pi4tLifRPnjyha9eufPDBB2zbto2LFy8yceJEbGxs6Nevn8bKKTqk/yd8fHzIy8tj+fLlaGkV3Dr8+++/v5H39vDwoGHDhvzyyy/s2LGD1atXl5mnevXqVK9enU8++YTBgwezceNGlR3SwMBA+vXrxxdffEHfvspL3Tdo0IDdu3dja2uLqalpuco6a9Yspk1TXrX1bq2h5cpblDw3j/Q7QZi2rE/S0cJFFkxb1ifpn9dfdMGiRwu09HRJ2Hv2tY/1IrnpWSSlZyntS4tNwrVlXWLvFaz+p6WrTaV3a3Lu692lHifyRiCuLeviu+GoYp+btwcRvoWdP10DPeQy5aumMpkMiVbhlcXnnVGLynbsHviVyqmgryM7PYvsYvEmxyZSs4Un4feCAdDW1cb93doc+HrHC4/V7sMedJ7clx9HfkXoncdqLWd55aZnkVwsnvSYJFxa1iW+SP05vVuTi0tLr7/oG4E4t6yL36+F9efi7UG0b+mj81p6OlhWcyJSxeNIag9qRcztx8Q/UF8nXZabz9PbT3Dwrkt4kUcKOHjXJbyURwjE+QZQqUMDpX0OreqScOsJ8rzyLxSio1/yuyvPl5V6W8Kreh6jvXddworEaP+CGON9A3BSQ4wAjb4cgXPnhhzv/yXpYepZiOr/wb/p7/Dh3ouEFbs3sde2GTzce5H7v5976djKIs/NJ/n2E2xaeRJdZHq+TSsPoo+q/s4m+gZg11H5O2vT2pOkW49LfGcdejZBS0+H8L0X1F7253LTs8hVUX+VitWf47s1ufyC+ot5Vn+3i9Sfc5H6C79wj13tZyrlabv8QxIDI7m57hBymZyHey+WuLe0+7YZPNp7EX811l9eehapxWLOiEnC0bsuT4vEbNekJr5flR5znG8gTi3rcv+XwpidvD2I9dHczK5/o4ocD16xYgVjxoxh7NixAKxcuZJjx46xbt06li5dWiL9+vXrcXFxUSxWWqtWLXx8fPjuu+802iEVixr9n6hatSp5eXmsXr2ax48fs3XrVtavX//G3n/s2LF8/fXX5OfnlzrSCZCZmcnkyZM5c+YMISEhXLx4kevXr1OrVi2VaXv06EH9+vX58MMPiY6OVmxQsDiStbU1vXr14vz58zx58oSzZ8/y8ccfEx6u+r4nqVSKqamp0vaq03Vjfj6A9eD2WA1sh757JZznj0bPyZq4rQX3fTjNHIbbyo+V8hjUroxB7cpoGeqjY2WKQe3K6FerVOLY1oPak3TsKvlJJVdP1LQbG47y7qSeuHdqiHX1SnRZPo68rBwe7C989EyX78fR8vP3CvP8dgw3bw8aT+iOZVUHGk/ojkuLOkod1KATN2nyUS+qtK2PaSVr3Ds1pOHYLopnlUq0tei5fgp2npU5PGUdEm0tDG3MMLQxQ0v31RdnKcvp347QcVJvPDs1wqG6M8O/m0huZjY+Bwr/ARq+fBI9ZwxW/N5+XE+6fzqQ7TPWkRAei4mNGSY2ZugZFj4iQ89QilNtV5xqF8wWsHK2xam2KxYaXDEYwG/DURpO7kmVzg2xrFGJ9ivGkZuVw6Mi9dfh+3E0LVJ/fhuO4eLtQYMJ3bGo6kCDCd2p1KKO0j/GzecMxrFJTUydbbCrX5Wu66egZ2yAf7Hp1rrGBrh3a8z9nWfUHpv/z39TdUhrqgzyxtTdkQYLhmLoZEXAlpMA1J/1Hk1/GKdIH7DlFEaVrGgwfyim7o5UGeRN1cGtebD+iCKNlq42FnVcsKjjgpauDoYOlljUccHYrXDkN/z4TepO6YVju/oYVbKmUueG1BzXRanTqC4Pfv4b9yGtqfosRq8FQzEqFmOzIjE+2nIK40pWeD2LseqzGO+/ZIyNvhpF5b7NuTBpLblpWejbmKFvY4a2vuoF5ypCRkYm/o+C8H9UMLITERmD/6MgoqLLd4vJm1RRf4dZSWk8fRiutMly80mPSyLpsWaeKfv4p8O4DGmD8+DWGFdzpM7C4Rg4WSueK1rzi0HUXz1BkT54ywkMKllTe8EwjKs54jy4NS6D2/B4XclnFbsMbkP0UR9yX2G0/3Xc3nAUr8k9qfys/tquKDgPBhSpv3bfj6NJkfq7veEYzt4evDOhO+ZVHXjnWf3delZ/uelZJeomNyObrMSCOgPILqX+MjRYf8/d//Uonh/1xKVzQ8xrVKLF9+PIz8wh6M/CmFv+MA6vmYUx399wDMdWHnhM7I5ZVQc8JnbHsWUd7hX5zuoYSrGs44JlnYLROmMXGyzruGCk4XPhmySTqHcrr5ycHHx9fUusmdKxY0cuXVL9uMDLly+XSN+pUyd8fHyUZlSqmxgh/T9Rv359VqxYwTfffMOsWbPw9vZm6dKljBgx4o28/+DBg5k6dSpDhgxBX7/0Z1Zpa2uTkJDAiBEjiImJwdramr59+5aYRgsQExODv78//v7+ODo6Kr0ml8sxNDTk3LlzfP755/Tt25fU1FScnJxo165duUdMX0fiwYvoWJjiOHUgurYWZD4MJWDEYnIiCkYWdG0tkTopTx2u88/3ip+N6rlj1acV2WGx3Gn6oWK/tLIjJu/W5tHg+RqPQZVr6w6ho69H+y9HoW9qSJRfEH8M/UbpCrKpo7XSiFGkbwCHJq+h+WcDaP5pf5JCYjg0aQ3RRaZBnZy3hRaf9af9klEYWJuSHpPIre2nuPzDnwCYOFji3tELgJHHvlIq0+73viTsygONxHti/V/o6esxcPEYDM2MCPYLZM3wr5RGUi2drJDLC6extxzeAV2pLmPXf6p0rCMr93BkZcEK1a6eVfl4V2Ed9ps7EoArf5xh22fr0JQbz+qv9ZJRSM0MifEL4kCx+jN2sla6xyfaN4CjL1DLrwAA0FJJREFUk9bQdPoAmnzWn+SQGI5NXKM0DdHYwZJOayZhYGFC5tMUom8E8nuv+aRGJCi9f/WeTUAi4dEB9S74AxDy11X0LEzw+KQPBrbmJD0M58ywb0l/VgZ9W3OMiiy+lR4Wx+lh3+G1cBjVR7UnMyYR37lbCDtyXZHGwM6CrscLv2+1J3Sj9oRuxFx6wIn+XwLgM2cL9Wb0p/HSUUitTMmMSSRw6ynufP+nRmKUFovxdJEYDVTEeGrYdzQsEqOPihi7lRLj8Wcx1hjVHoCO+5TvXbs09Sce/66B+2RfwV3/AEZ/9Lni92WrCxa56dWlPV/O+bS0bBWiov8O36TIA1fQtTCh+rS+SG3NSfUP4+rQb8gMjwdA384cgyLf2czQOK4NXUadhcNxe78j2TGJ3J2zmajDyo/IMqpij1WTmlx+T/l88CbcfFZ/3kXq72A56u+fSWt4d/oAGn/Wn5SQGI5PXEOsn2bufVW3O2sPoa2vR9OvRqFnZkj8zSCODfmGvCIxGxU798f6BHBm4hoazBjAO9P7kxoSw5kJa4gvMh3Yul4VuvwxW/H7uwuGARDw+zkufPJmFqr6r1G13olUKkUqVX4udHx8PPn5+djZKd86Y2dnpxi8KS46Olpl+ry8POLj43FwcFBDBCVJ5P9vdxYLFSIsLAw3NzeuX79OgwYNys7wL+JTqXdFF0GjzmgZl53oPyxY680uJFQRauT/e0apNMHyzTxmr0K97SfigbcXVXQRNGr9O6Wvivq2cHtDz7usKGEanGnzb2Dwtj7Asoj3I7ZVdBFU2uQ0TK3HC/7AvcRAzfz581mwYIHSvsjISJycnLh06RJNmzZV7P/yyy/ZunUr/v7+JY5dvXp13n//fWbNmqXYd/HiRVq0aEFUVBT29vZqjeU5MUIqaFRubi5RUVHMnDmTJk2a/Oc6o4IgCIIgCILwqtR9wVHVeifFR0cBrK2t0dbWLjEaGhsbW2IU9Dl7e3uV6XV0dLCy0tw0anEPqaBRFy9exNXVFV9f3zd6z6ogCIIgCIIgvG1UrXeiqkOqp6eHl5cXx48fV9p//PhxmjVrpvLYTZs2LZH+n3/+oWHDhiWeuqFOYoRU0KjWrVv/3z1vShAEQRAEQRDg5RYiUrdp06YxfPhwGjZsSNOmTfn5558JDQ1VPFd01qxZREREsGXLFgDGjx/PmjVrmDZtGh988AGXL19mw4YN7Nyp2ccMig6pIAiCIAiCIAiCBlTk7bsDBw4kISGBRYsWERUVRd26dTly5AiurgUr/EdFRREaWvgItsqVK3PkyBE++eQTfvzxRxwdHVm1apVGH/kCokMqCIIgCIIgCILwVpo4cSITJ05U+dqmTZtK7GvVqhU3btzQcKmUiQ6pIAiCIAiCIAiCBvwfLHD82kSHVBAEQRAEQRAEQQPkFXgP6X+FWGVXEARBEARBEARBqBBihFQQBEEQBEEQBEEDxJTdsokOqSAIgiAIgiAIggaIDmnZxJRdQRAEQRAEQRAEoUKIEVJBEARBEARBEAQNkFd0Af4DRIdUEMrgLzOu6CJolP5bvvpb+yztii6CxpnKcyq6CBqV9X8wmUfO2/2HuP6deRVdBI0af3NRRRdB4xY0nFPRRdCoPPIruggaZaD19rej/1ayt7t5Vwvx7RQEQRAEQRAEQRAqhBghFQRBEARBEARB0ACxqFHZRIdUEARBEARBEARBA0SHtGxiyq4gCIIgCIIgCIJQIcQIqSAIgiAIgiAIggaIVXbLJjqkgiAIgiAIgiAIGiBW2S2bmLIrCIIgCIIgCIIgVAgxQioIgiAIgiAIgqABYlGjsokOqSAIgiAIgiAIggaIe0jLJqbsCoIgCIIgCIIgCBVCdEiFcmvdujVTp06t8GOUZtSoUfTu3VsjxxYEQRAEQRCElyVDrtbtbSSm7AoacebMGdq0aUNiYiLm5uaK/fv27UNXV1fxu5ubG1OnTtVYJ1WTqo9sT+0JXTGwNSfpUQQ+87YRd+1hqeltm9TEa8FQzKs7kRGTxP21hwjYekopjXPXRtSb0R8TV1tSQ2K59fUewo76KKUxsLegwexBOLbxRNtAj5TH0VyZ9gtP7wQr0pi6O9JgziBsm9REoiUh6WEE58evJiMiQS2xN/6kL3WGtkFqZkTMzSDOztnE00cRL8xTtUsj3v2sP2autiSHxHLl2z08Lhbbc16TetB05kD8fj3KhYXblF6zcHek2ReDcHy3ILanjyI4OmE1aZHqia3yqPa4T+yOvq05qQ8juDNvCwlXS69Xq6Y18VgwHJMaTmTFJBHw40GCt5xUvO4y0JsGP4wvke8v15HIsnNL7K/2UU/qzB5E0M9/c2feVrXE9CJOozriMqknerbmpD8MJ2DuJpKv+qtMq2drjvvCEZh4VsGwij3hv/5NwNzNSmmMalSi8oyBmHhWxsDFlkdzNxH+8xGNx/Gcy6gOVJnUA6mtOWkPw7k/dwuJpcQDYNm0FrUWDse4RiWyYxJ5vOYgoVtOKF5/d988rJrXLpEv9vgNfIYtA0DbSJ/qM9/Dvksj9KzNSLkbzP05m0j2e6z+AAHXUR2oMqm7IsZ7c7eQ+ILvqGXTWtReOEwRY9CaQ0oxNtk3t5QYb3L9WYwuI9vjOqoDBs7WAKQ9DCdg+T7iTt1Sc3QFnrcx+mZGRL9EG9OkSBtzuVgb0/iTvrw7ra9SnvTYJH7zmqzyeG2WjqbusLacW7CVWxuOvX5QauDjd4eNO/7gvn8gcQlP+WHpXNp5N6voYpVb26n9aDS4LQZmRoT5BXJw7kZiA0qvV9tqTrSbNgAnj8pYVLLh8KItXPrtaKnpvSf2pNOMQVz87W+OLNJ8+1lc+6n9eHdwOwzMjAj1C+TA3I3EBISXmt6uWiU6TOuPk0cVLCvZcHDRFi789neJdKZ2FnSZOYQareuhq69H/JMo/pjxMxF3n2gynBJaT+2L15C26JsZEXEzkMNzNxH3gvqzqeZEm0/741i3MubONhxduJUrxeqv4bB2NBrWHvNKNgDEBoRz9oc/CTyjmbalIoh7SMsmOqTCG2VpaVnRRVAL157v4rVwGNe/2ETstUdUG96Wttunc7D15yo7fUbONrTd9hkB289wcfI6bBtXp9FXo8hKSCXsyHUArL3cabl+MreW/UHYUR+cOzek5U+TOdZ7MQk3gwDQMzOk04F5xFx6wKlh35IVn4KJmx05KRmK9zJ2taXT/rkE7jrLre/2kpuSgVk1J/KzSnZ+XkWDCd2p/0EXTkz7iaQn0TSc0oteO2ayrdV0ctOzVOaxb+BOp7WTufrdHwQd9aFq54Z0WjuZfX0XE+MXpJTWtl4V6gxpQ/z9kBLHMXW1pd++udzfdZary/eSk5qBhbsT+So6dq/CqVcTPBaN4NbM30i4/ojKw9vRdMfnnPSeTqaKejV0saHp9hmEbDuNz+QfsWpUnXpfjyYnIYXIw9cV6XJTMjjR/FOlvKo6o+b1q+A2vC3J90rGrgm2vZpSbfEoHs78leRrD3Ea0Z56O7/gastPyFYRr5ZUl9yEFEJW7sN5XDeVx9QykJIZEkPswctUWzRS0yEocejVlNqLR3J35gYSrz3EZUR7Gu2cybmWn5KlIh4DFxsa7vicsG2n8Ju0BovGNaj79RhyElKIPnwNgBujlyPRLTxV6lma0OLUN0QdvKrY5/H9OExqVMJv8o9kRyfi1L8ljffM4VzLT8mOTlRzjE2ovXgEd2f+poix8c6ZnG35WakxNtoxg7Btp/Gb9OOzGEcrxeg7egVaRWLUtTSh5amviTp4RbEvK+op/kt2kvEkGoBKA71puPkzzrefRdrD0v/hfhUNJnTnnWdtTOKTaBqVs43pvHYyV777g8dHff7H3n2HN1X1ARz/pk2a7nQPSqEUkFlA9t57LxXZgoosFQQVZMPLcqGAggtEhgMUZRVkCsiGtmxK995Nd7ry/tGSNm06oA2Fej48eR5yc+7N+fXee27OPePi3r81/b+axb4iZUzc/RD2v7pW8z43R/dPRfd+rXB8sS4pkfGVGltFpadn0KCeO8MH9mXOR6uqOjuPpctbQ+g0dQD75m0lNiCCHrNH8NrOhXze8z0yS9ivMhM5CcHR3Dp8iUGLx5e6fZdm7rR5tScRd59O+VlUt7eG0GXqQH6dt4XYgAh6zR7B6zsX8nHPuaXEZ0R8cDQ3D19i8OIJOtOYWJoxfd9y/C/c5ofJ60iNU2JTy5H0pFR9hlNMp7cG0+H1geyft4U4/0i6zh7OxF0L2NhjXpn7786hS/Rbonv/JUXEc3zdz8QHRgHQfHQXXv12LlsGLiy1sitUL6LLrvBEdu7cSevWrbGwsMDJyYmxY8cSHR0NQGBgID169ADA2toaiUTC5MmTAe0uu927dycoKIg5c+YgkUiQSPIe1LRs2TJatGih9X0bNmzAzc1N8z4nJ4e5c+diZWWFra0t77//Pmq1djcGtVrN+vXrcXd3x8TEhObNm7N3795Kib/RmwPw23Oah7tPk/QwnGtLd5IWHscLE3vpTP/CxJ6khsVxbelOkh6G83D3afx+PkPjtwZq0jR8oz8R/9zi9qYDJD2M4PamA0Seu0OjN/pr0jSeOYS08HguzPmGOC9/UkNjiTx3m5SgaE2aFh++RNhJb26s+pmEW0GkBMcQdsILVVxSpcTefGp/rm78E3/Pq8TfD+X4nK1IjY14YXjJd+mbv96fkLO3uLb5AIl+EVzbfIDQ83do/np/rXQyUzl9v5zOyQ++R6VMK7ad9u+/ROBJb/5d/TOxt4NICo4h6KQX6ZUUW91pAwnac5qg3adJ8Q3n5pKfSA+Lo86k3jrTu03sRXpoHDeX/ESKbzhBu08TtOc09aYP1k6oVqOKUWq9ijI0ldN680y83vuOLOXT+aHh+tZgwnefJGLXSdJ8w/Bd/COqsFhcJvfVmT4jJAbfRduJ/O0fspOK7x+AZC8//FbsJHr/vzor3fpU561BhOw+ReiuU6T6hnN38Q4ywuKoPbmPzvS1JvYhIzSOu4t3kOobTuiuU4TuOUWdGQX7LysxlcwYpeZl182DnHQVkfmVNQNjGU6D2nJv5W4SLt4jLTAK30/2kh4cXeL3VkaMIbtOkeIbzp0yYqw9sTcZoXHcWbyDFN9wQnadImTPadxnFNxQyEpM1To2H8VYuNIdfew6MSe8SPWPJNU/kvtrfiU7NQPrVvUqPcYWU/tzZeOf+OWXMX/P2YqsjDKmRaEyJqFQGdOiSBmTm51LWoxS88qITy62LTMna7qtnMSxt78iNyun0uOriC4d2vD2m5Po071TVWflsXWa0p/Tm//kztErRD8IZe97XyMzMaL5sJL3a5iPP55rdnPzwAWyM7NLTGdkKuflDTPZ/+F3pD+l8rOozlMGcHLzfm4fvULUg1B+yY/vxWEl76tQH38Or9mNdynxdZs+BGV4HL/N30qotx8JobH4/Xub+OBonen1pf3U/vyzaT93Pa8S/SCUP97bgszYCI9S9l+4jz9/r97DrQMXyVHpju/BiRv4nvImLiCSuIBITn78G5lpGdRsWfllS1VRV/KrOhIVUuGJZGZmsnLlSry9vdm/fz8BAQGaSqerqyv79u0D4P79+0RERPDFF18U28bvv/9OzZo1WbFiBREREURERJT7+z/99FN++OEHvv/+e86dO0d8fDx//PGHVppFixaxbds2vv76a27fvs2cOXMYP348Z86cefLAAQOZITbN6hBx5pbW8ogzt7BvXV/nOnat6hdPf/omts3rIJEaAmDfqh4RZ25qpQk/7YNdoW3W7NuSOG9/umydzWifzQw8top6Y7sXrCCR4NKrBcn+kfTc/T6jfTbT/+AyavZvVYGIC1jWssfM0YrgfwrymZuZTdilezi30h075LVeFF4HIPiMD05F1um2ajKBJ70IPXe7+EYkEtx6tiAxIJKhO99nyo3NjP5rGXX6VU5sEpkhVs3qEH3aR2t59Jmb2LR5Qec6Nq3qE11kn0Wf9sGq0H6FvC6dfa9+Qb/rG2n/0zwUTWsX21bzta8RefwGMWdvFftMHyQyQyyauRNfpFtU/BkfFK0bPJU8VCaJzBDLZnWILbL/Ys74YNVa9/6zbl2fmDNF0p/yQdHcXWv/FeY6tgcR+y+Qk6bK+15DQwykhsUq3zkZmVi3bfik4egkkRmiaFaHGB0xWpcQo5XOGL3LiLG7VozFGEhwHt4BQ1M5CVd9Hz+QUui7jLGq48hrVzcy8fxn9Ns8E8ta9tobkkjos+Etrm85VGYXYaH8rF0dsHCw5uHZgmMxJzObwEt3qdVK97H7OIasfI37p27gd/7plJ9F2bg6YOlgje/ZgmMwJzMb/0t3qV3B+Br3bkXoTX/GbX6HxVe38PahNbQd07OiWX4s1q72WDhY41ckvsBL93At5bx8XBIDCU2HtEdmIif0+sNK225Vy63kV3UkuuwKT2TKlCma/7u7u/Pll1/Stm1bUlJSMDc313TNdXBw0BpDWpiNjQ2GhoaaVtbHsWHDBhYsWMCoUaMA2LJlC0ePFozxSU1N5bPPPuPkyZN06NBBk89z586xdetWunXr9ljfV5jcxgIDqSHpsdqtXOkxSmo4WOlcx8ReQXiRVrH0WCUGMinGNhakRydibG9FRpFtZsQqMbFXaN5b1LLHYmIv7n7jya2Nf2HXoi6tV04kJzObgL3nMLazRGZuQpNZg/Fat5cb//uZGj2a0+27d/h79GqiL5Y8lq48TO2tNHkvGrtFTbtS1yu2TqwSs0Kx1R/aHnsPN34dvET3NuwsMTI3odWMwVz8eC//rv6ZWt2bM/Cbd/jjldWEVzC2R/u1aOulKkaJvFA+CzN2sCI6xqdYegOZFCMbC1TRiST7hnP9nS0k3Q1BZmGC+xv96fLXMk71WkBqfvdHl2EdUHi4cab/4grF8DhkNpYYSA3JLBJvZowSoxKO42eZUX48RfdfZowSeQnxyB2sisVfdP8VpnixLhaNauEzZ6tmWU5qBglXHlBvzkhSHoShikmkxohOWLWsR6p/ZKXE9ohRCftMFaNE7qD7GJU7WOn8m5QWo2WjWvjM+abYtiwaudLx0AoM5DJyUjO49tpnpFRypa2kMiatHGVMWtF1ipQxUTce8ve7W0kMiMDUTkHrt4cz+o+l7O71IRmJKQC0mjEYdU4u3j88G2NGqwuL/P2QUuRYTIlJwqqU/VoeHkM6UKOJG18Pe3rlZ1GP4ksuFp8S6wrGZ1PLgfbje3P2u8Oc+upPXJvXZeiySWRnZnH997MV2nZ5meeXoalF4kuNVaJwqVh8AA4NXHn9j2VI5TIyUzP4Zdrnorvuf4yokApP5MaNGyxbtgwvLy/i4+PJzc27ZxMcHEzjxsUnx6hMSqWSiIgITUUTQCqV0rp1a0233Tt37pCRkUGfPtrd2DIzM3nxxRdL3LZKpUKl0m4VyFLnIJPoaEko0kVYIqFYt+Gy0uctLrS82OoS7fUMDIj38cdr7a8AJNwKQtHAhRcm9iJg7zkkBnkbDTl6nXvf5k0ckHA7GPvW9XlhYq/HrpC+MLwj3dcW3Hw4OPkTXaGARFJ8WRHFP5doYjd3tqHLsgn8OW5dieNBH8UWcOw63t/lxRZ7Jxjn1vVpOr5XhSukJeZTQql9ZIrt84IdC0DC9YckFLrTG3f5AT3+/h/uU/tyc9EOTGrY4LFqIv++suapd3HNz6j2Wwm6dtZz5DHjKc95mc91bA+S7wajvKE97tl75mY8Nkyjl8/X5GbnkHQzgPDfz2PpUedJAihT8dNPUkaMxVbIX64rxu4k6YgRIOVhOGd7fohMYYbT4LY0/3I6F0esqFCl9IXhHelRqIw5UEIZkxdj6dsqrYwBCCrUshxHKBHXHjLx3Kc0fKkLXt8ewd7DjeZT+vHLwEVPEopQSPNhnRi2eqrm/Y4peZNjFT8Wy7hulkHhbMPgJRPZNnEN2U+x/GwxrBMjV7+ueb9NE1/R8qTsa2NZJBIDwm76c/TjXwAIvx2IY/2atB/fW28VUo/hHRlSaP/teu1jQNdPlDLKnnKK8w9ny4CFGFua0mhAW4Z/+hbbX1lVbSqluZKqzsGzT1RIhceWmppK37596du3Lzt37sTe3p7g4GD69etHZmZmhbdvYGBQ7AKVlfV4F5pHFeRDhw7h4uKi9ZlcLi9xvTVr1rB8+XKtZSPMPRhp0UzzXhWfTG52Dib5d/IfMbZTkKFjbCDktSCaFGmlMbZVkJuVjSoh7858RkwixkVa4oztLEmPLRgfmR6diPJBuFYapW84tQa2KchbVjbKIj8Qlb5h2Ld9/G6YAX9f15oQxNAor8gwtVeQVqhlxcTOkvQSYgdIi0nEtEhsJnaWpOXHZt+sDqb2Cl45vFLzuYHUkBrtGtBsch++rjuZ9PhkcrKyiS9ygYr3DaNGm4p3MX20X42LtDTJ7RSoYnXHlhGdiHGR/Sq3syQ3K5vM/P1ajFpNgpc/5u55vQKsmrljbK+g+7H/aZIYSA2xbd+QOlP68letiZBb+RXErPgkcrNzMCpyHBvZKYq1wD0PMvPjkeuIR9eYXQBVdGKx1mAju7zzMqvI/jMwMcJ5eEd81/9WbDtpQVFcGrECQ1M5UnMTVNGJtPjmHdIreYxXQYzax6iRnSWqGN3jqFXRicVaT41KOEYNTIyoMbwjD3TECKDOyiEtf+IRpbd/3iRcb/Tn1vzvnzSkxypj0sooY8yK/F1MC5UxumSnq4i7F4JVHUcAarRtgKmdJZMvFgwxMZAa0nnxOFpM7c+PHec8Vmz/ZXePXyPEq+BGnDR/v5o7KEiOSdQsN7ezJKWE8rU8ani4Y26vYMaBgvLTUGqIW9uGtJ/Yl6UvTESth/LzTrH48p4eYOFgpRWfWQXjA0iOTig2U2+0XxhNB7St0HZLc//v64TdKH5emtsrSCl0XprZVjw+gJysHOKD8sqW8JsBuDR3p91r/Ti48IcKb/tZUF0f1VKZRIVUeGz37t0jNjaWtWvX4urqCsDVq9qP7zAyMgLyJh8qjZGRUbE09vb2REZGolarNRMdeXl5aT5XKBQ4Oztz8eJFunbtCkB2djbXrl2jZcuWADRu3Bi5XE5wcPBjdc9dsGABc+fO1Vq2r8E0rfe5WTnE+wTg1LWp1iNZnLo2JfToNZ3bjb3mi0ufllrLnLs1Jc47AHV2Xvwx1x7i3LWppmUzL40HsYXGaMVceYBlXWet7Vi6O5EaFqvJW5y3f7E0Fu7OpIbGlhq7LlmpGSiLzJ6XGpWIa5emxObPBGsgM8SlXUP+XfNLiduJvP4Q1y5NNS2bALW6ehB5LS+20HO32d37Q611en36JgkPw7n+9UHUuWrUuTlEe/tj5a4dm5W7M8lhjx9bUeqsHBJ9ArDv5kHEkYL9at+tKZGeuvdr/DVfnPpq71eH7s1ILLRfdVE0rU3S3RAAYs7e4kT397U+b7lhGim+4TzYfEAvlVHIizfZxx+bbs2IPVIwI7BN12bEHL1SyprPJnVWDkk+Adh18yCqUDx2XT2IPqr78UIJV31xKLL/7Lo3Q+ntX2z/OQ/tgIGRlLC9JbdI5KSpyElTIVWYYd+9GfdW7q5ARMWps3JQ+gRg360ZUYWOUbuuHkSVUPYk6ojRvoQYawxtnx/jufJlSCLBwEhWdrpSlFTG1NJRxpwvRxnjVUIZo4uBkRSb+i6E5z+u6/6+84QUGb8+bOf73N93nju//vPYsf2XZaZmEF9kvyZHJ1CvswcR+fvVUGaIW7tGHF2754m/x+/8Lb7oq11+jvp4GjF+4fyz5YBeKqOQF19ckfiSohOo39mD8NuBQF587u0acaQC8QEEXnuAvXsNrWV2dZxJrITrXklK2n91O3sQqbX/GvL32p8rPwOSgkq+8N8gKqTCY6tVqxZGRkZs3LiRt956i1u3brFy5UqtNLVr10YikXDw4EEGDhyIiYkJ5ubmxbbl5ubGP//8w5gxY5DL5djZ2dG9e3diYmJYv349o0ePxtPTkyNHjmBpaalZ75133mHt2rXUr1+fRo0a8dlnn5GYmKj53MLCgnnz5jFnzhxyc3Pp3LkzSUlJ/Pvvv5ibmzNpku7HUcjl8mItqLq669795ggdv5xOvI8/MVcfUn98D8xcbPHNf/5kiwUvY+pkzb/v5I01e7DjJA1e60OrpePw3XUK+9b1qPtqd87N2KzZ5r3vjtL390U0njmY0KPXqNmvFc5dmnB0eMHf9t43nvT7awlNZg8l6MAl7F50p/74HlycX3AX8c5Xh+m8ZRbRF+8R+e9davRoRs0+L/L36II7yBXh/b0nrWcNRRkYlffYl1lDyc7I5MH+fzVpen8+jdTIBC6s+zV/naOM3LuIltMH43/sGu59W1GzcxN+H5kXW1ZqBvFFHh2RnaYiIyFFa/mNrYfpt3kW4ZfuEXbhLrW6NaNO7xf54+XKic1v62FabZxBorc/8Vd9cRvfE1MXOwLy92vjha9g7GzD9dlfAxC44wTuU/rSdNl4AnedxKZ1fWq/2p2r0zdqttngvZEkXHtIin9k3hjS1/uhaFIb7wXb8+JMzSD5nnbsOWkqMhNSii2vbCFbDtJ402ySvf1RXn1AjQm9kde0I/zHvwFw/+hV5E423J1dcJyaN8mbkMnQzBiZrSXmTWqTm5VNWn6rvERmiNkLNYG8H/xyJxvMm9QmJzWD9PzWNX0J2HKI5ptmovT2J+HqA2pN6I1JTTuCfsx75maDj8Ygd7LBZ/ZXAATv+JvaU/vSaPkEgneewLr1C7iO7YHXW18W27br2B5EeV4t1nIKeZVYJBJS/cIxc3Oi4dJxpPpFELrntF5ibLFpJone/iRefYDrhF6Y1LQjuFCMxk7WeOcfo0E7jufHOJ6QnSexyo/xxlsbi227tBgbLHyF6BNeZITHITU3ocbwDth2bMzlMWuLpa0or/wyJrFQGZNVpIzp8/k0UgqVMV7fH2VUfhkTcOwadfLLmH0jC8rPToteJeD4DVLC4jCxtaTN28MwMjfhXv5NhozEFM1Y0kdys3JIjUkk0b/8k+7pU1paOsGhBb1kwsKjuPfAD4WlBc5ODlWYs7Kd/8GTbjOHERcYSWxAJN1nDiMrPRPvPwv26+hPp5MUFc+x9Xk3HwxlhjjUr5n/fymWjjY4N66NKjWD+KAoMlMziH6gXU5mpqtIS0wptlzfzv1whB4zhxEbGEFsQCQ9Zg4nKz2TG3+e16R5+dPpJEUl4Ln+5/yYCuKTyqRYOlrj3Lh2XoU3v8Xw3PeHmbFvOT1mDMPn0EVcm9el3as92bfgu6ca38XvPekycyhxgZHEB0TSZdYwsjIyuVlo/4347C2SIhM4UWj/2T/af0ZSLJysccqP71GLaK/5L+N72pukiDiMzExoOrQ9bu0bs3Piuqcanz6J9tGyiQqp8Njs7e3Zvn07Cxcu5Msvv6Rly5Z88sknDB06VJPGxcWF5cuX8+GHH/Laa68xceJEtm/fXmxbK1asYNq0adStWxeVSoVaraZRo0Z89dVXrF69mpUrVzJq1CjmzZvHN98UTLLx3nvvERERweTJkzEwMGDKlCmMGDECpbKg68jKlStxcHBgzZo1+Pv7Y2VlRcuWLVm4cGGF/wZBf11Cbm2Bx5wRmDhYkXg/lFPjPyY1/zmAJg5WmBUa6J8aEsPJ8Z/Qevl4Xpjcm/SoBK4u3qF5BilA7FVfzk3fRPMPXqL5/NGkBEVx9q1NmmeQAsR5+3Nm6gZaLHiFZnOGkxISw9UlOwn8o+CCEOJ5lcsf/kCTWUNpvXIiSf4R/PPGF8RcflDhuAGuf30QqbER3VZNRq4wJcrLjz/HrdN6PqCFi51Wt+vIa74cnbmJ9vNfot280SiDojg6Y1OxZ5CWxd/zKqcX/kCrmUPpumIiCX4RHJn2BRFXKie2sD8vYmRtTsO5I5E7WJF8L5QL49aTnt+6bOxohamLrSZ9WnAMF8atx2P5BOq81oeMqAR8Fv2o9QxSmaUpLT6ZitzeiuzkNBJvBnF2+EoSdYzRe9qi/7yAzNoCt7mjkDtak3IvBJ+xa8jIj1fuYI1xkQkr2p78WPN/yxZ1cRrVhfTgaC60mZW3jpONVpraM4dSe+ZQEs7f5sZI7e7wlS3izwvIrM2pN3cUckcrUu6FcGXsWq14TArFkx4cw9Wx62i0YiK1XuuLKiqBOx9t1zyf8xEzd2ds2jfk8ku6b3xILU1p8NGrGDvbkJWYQuTByzxY83OpreRPHuNFjKwtqD93ZKEY12mOUbmDVbEYr4xdT+MVE6idH+Ptj37UEaMTNu0bcuml1Tq/V26voMWmmcgd847j5DvBXB6zltgiM9tWhkdlTPdSyhhzHWWM58xNdJj/Eu1LKGPMnW3ot2kmJtYWpMcnEXn9Ib8OW0qyjue3Pqtu3fNlyuwPNO/Xb8y7Lg4b0Jv/LXqvpNWeCWe3HEBmbMTQla9hrDAj1MuPbRPWaD3DUuFii1pdMI+ohaM1sw6v0bzvMm0wXaYNxv/iHb4f82w9h/VMfnzDV07BRGFGiJcf301YrRWfVZHj1tLRmncPF9zU6TZtCN2mDcHv4h2+GZN3MyXUx58d0z6j//tj6PXOSBJCYjiw4ie8ClV0n4bzWw4iMzZi0KrJmFjm7b+fxq/V3n81bLVapS0crXnrSEGZ0mnaYDpNG0zghTtsH5NXnprZKxj5+XTMHaxQJacRdS+EnRPX4X+uamZM1ofqOjNuZZKoKzKaXBD+A3bWKP1h3M+7RN1Pfqg2amZV/0uBpbrk5/NVBxn/gSeUqanes174G1XvguatGyuqOgt6t6x19Z7sKbuat2OZqKt/ObosaFdVZ0GnBW5jK3V7awIrdzjIs0C0kAqCIAiCIAiCIOiBmNSobKJCKgiCIAiCIAiCoAeiOlq26t9+LwiCIAiCIAiCIDyTRAupIAiCIAiCIAiCHlT/mSwqTlRIBUEQBEEQBEEQ9ECMIS2b6LIrCIIgCIIgCIIgVAlRIRUEQRAEQRAEQdADdSW/9CUhIYEJEyagUChQKBRMmDCBxMTEEtNnZWXxwQcf4OHhgZmZGTVq1GDixImEh4c/9neLCqkgCIIgCIIgCIIe5FbyS1/Gjh2Ll5cXnp6eeHp64uXlxYQJE0pMn5aWxvXr11m8eDHXr1/n999/58GDBwwdOvSxv1uMIRUEQRAEQRAEQfiPunv3Lp6enly8eJF27doB8O2339KhQwfu379PgwYNiq2jUCj4+++/tZZt3LiRtm3bEhwcTK1atcr9/aJCKgiCIAiCIAiCoAfqSu5oq1KpUKlUWsvkcjlyufyJt3nhwgUUCoWmMgrQvn17FAoF//77r84KqS5KpRKJRIKVldVjfb/osisIgiAIgiAIgvAcWLNmjWac56PXmjVrKrTNyMhIHBwcii13cHAgMjKyXNvIyMjgww8/ZOzYsVhaWj7W94sKqSAIgiAIgiAIgh5U9hjSBQsWoFQqtV4LFizQ+d3Lli1DIpGU+rp69SoAEomk2PpqtVrn8qKysrIYM2YMubm5fPXVV+X/4+QTXXYFoQymudX7kcZx0up9XyqqmscHcKqax2jxH7h3KleXfcF/nrVQ5VR1FvRqWetFVZ0FvVt2dVVVZ0GvDjSt3vtQrq7ev2WeZZX9HNLH6Z47a9YsxowZU2oaNzc3fHx8iIqKKvZZTEwMjo6Opa6flZXFyy+/TEBAACdPnnzs1lEQFVJBEARBEARBEIRqx87ODjs7uzLTdejQAaVSyeXLl2nbti0Aly5dQqlU0rFjxxLXe1QZ9fX15dSpU9ja2j5RPqv/bWdBEARBEARBEIQq8Dw8h7RRo0b079+fN954g4sXL3Lx4kXeeOMNBg8erDWhUcOGDfnjjz8AyM7OZvTo0Vy9epVdu3aRk5NDZGQkkZGRZGZmPtb3ixZSQRAEQRAEQRAEPajsLrv6smvXLt5++2369u0LwNChQ9m0aZNWmvv376NUKgEIDQ3lr7/+AqBFixZa6U6dOkX37t3L/d2iQioIgiAIgiAIgvAfZmNjw86dO0tNo1YXVK7d3Ny03leEqJAKgiAIgiAIgiDogZhOqmyiQioIgiAIgiAIgqAH6ueky25VEpMaCYIgCIIgCIIgCFVCtJAKgiAIgiAIgiDogeiyWzZRIRUEQRAEQRAEQdAD0WW3bKLLriAIgiAIgiAIglAlRIX0P6x79+68++671eY7J0+ezPDhw/WybUEQBEEQBEF4XLmV/KqORJdd4an6/fffkclkmvdubm68++67T71irC/uk3tTf8ZgjB2sSLofhs+SHcRdul9iersODfFYNgHLBi5kRCXyYPMBAnac0Jm25rAOtN06m/AjV7n42mea5f2ufIGZq32x9H7bjuG9YHuFY9Kl3ZyRNB3bA2OFGZE3/Di1eDvxD8JKXafegDa0nzcaRS0HlMHRXFj/G35Hr2pts/2ckVrrpEYn8l3rWZr3pnaWdFowhlpdPZBbmhJ26T5nlvxIYmBUpcbXau5IGo3tgdzKjOgbfpz7aDsJZcRXZ2Ab2swbjWVtB5KCorm8/jcCPQviazyhF40n9sKiZt6+SngQyrUNfxByygcAA6khbd4fjWvPFljWsiczKZ2wc7e4tOYX0qISKzU+Xfq/O5qOr/bERGFOkNdD9i7+gUjf0BLTO9WvycC5L1HTwx3bmvb8vuJHzvxwRCuN3MyYge+9TLO+bTC3UxB2O5Dfl28n2Mdf3+EU0/PdUbR+tScmCjNCvR5yYPE2on1L3qcO9V3oNfclanjUwbqmPYdW7ODCD54lpu86Yyh93x/Dvz8c4fCKn/QRQqm6vjuSlmN7YqwwI+zGQzwXbyemlPjs67vQ7b3RODetg5WrPUeX/8TlIvHVatuQDtMG4exRBwtHa3594zPuH7um71CKqT25D/VmDEbuYEXy/VBuL9lBfCnlqm2HRjReNh6LBjXJiErAb/NBgnYc13ze4ffF2HVsXGy9qOM3uDx+vV5iKI+e746iTf4xGvIYx6hLoWP03zKO0X7vj+F8FR2j5XHV6ybbdu/lzr2HxMTF88WaxfTq2rGqs1VMVVzrG80bRaN5o7TSZkQncrjZjMoJqpDak/vgPjPvnEu5H8rtxTtIKCU+mw6NaLx8POYNaqKKSsBv00GCC51z7X9fjG2n4udc9N83uJJ/ztWa1Jvak/tg4moHQMr9UHw//Z2Yk96VHN3Tl1tJz+qszkQLqfBU2djYYGFhUdXZ0AuXYe1ptmIi9zfs52SfhcRduken3R9g4mKrM71pLXs67nqfuEv3ONlnIfe/2E/zVZOoMahNsbQmNe3wWDqW2At3i312qv8iDnlM17zOvrQagLADlyo3wHytpg/mxdcHcHrxj/w8eAmpMYmM2PUhMjPjEtdxalmPAZtnce/3c+zuv5B7v59jwFezcGxRVytd7P0Qvm01U/Pa1XeB1ueDv52DopYDB6d+zu4Bi0gOi2XE7gVITeSVFl/zGYNp9sYAzi/+kd8HLSEtOpFBu0uPz7FlPXp/NYsH+86xt+9CHuw7R++vZ+HwYkF8qRHxXFrzC78PXMzvAxcTdv4O/b6fi/ULLgBITYywa+rG9Q372dd/Mcfe3IDC3Zn+P8yttNhK0uutofSYOpC9S7bx2dCFJMckMmPnQuSlxGxkYkRscDQH1u1GGZ2gM82YddNo0NmDnXM3s67ffO6d9WHGzkUoHK31FYpOXd4aQsepAzi4ZDtfD11EcoySyTsXYlRKfDITOfHB0Rxb9zPJJcT3iEszd9q82pOIu0GVnfVy6fjWYNq/PhDPJdv5fshiUmOUjNu1oNT4pCZyEoKjOVlKfDJTOVF3g/Fcsl1POS9bjWHtabpiIr4b9vNPnwXEX7pPu90flliumtSyp+2u94m/dJ9/+izg4Rd/0nTVJJwHtdWkuTrlM455vKV5neo2n9zsHMIPXHxaYRXT5a0hdJo6gANLtvPV0EWkxCh5rRzHaEJwNEefg2O0vNLTM2hQz52Fcyu/klVZqupaD6C8F6J1vT/e44NKjQ3AeVh7Gq+cyMMN+znXO++ca7vnQ4xLOefa7M475871zjvnmvxvEk6FzrlrUz7jeNO3NK8zXfPOuYhC51xGRDz3Vu3hfN+PON/3I+LO3ab1j/Mwb1Cz0mMUnj2iQioAkJCQwMSJE7G2tsbU1JQBAwbg6+ur+Xz79u1YWVlx9OhRGjVqhLm5Of379yciIkKTJjs7m7fffhsrKytsbW354IMPmDRpklY32sJddrt3705QUBBz5sxBIpEgkUgAWLZsGS1atNDK34YNG3Bzc9O8z8nJYe7cuZrvev/991EXuQOlVqtZv3497u7umJiY0Lx5c/bu3Vs5fzAd6k8bSOCe0wTuPk2ybzg+S34iLSwO90m9daavM7EXaaFx+Cz5iWTfcAJ3nyZwz2nqTx+sndBAQpvNM7nz8T5Sg6OLbSczLhlVjFLzcu7zIikBkcT+q/uCVlEvTu3PlU1/4ud5lbgHofw9dysyYyMaDC/5LvaLU/sTfPYWVzcfIMEvgqubDxBy/g4vTu2vlU6dnUtajFLzSo9P1nxmVccJ51b1OfnRNqJ8/En0j+DUR9uQmclpMKxDpcXnMbU/1zf+ScCRqyTcD+XUnK1ITYyoV0p8Hq/3J/TsLbw2HyDRLwKvzQcIP38Hj0LxBR2/QchJb5QBkSgDIrmy/jey0jJwaFkPgMzkdA6NXYf/wUso/SOIvu7H+cU7sG/ujnkN3T8EKku3KQM4tnk/PkevEPEglJ3vfYXMRE6rYZ1KXCfYx5+/1uzixoELZGdmF/tcJpfRvH9b/lqzG7/L94gNisJzw17iQqPpNL6PPsMppuOU/pzZ/Cd3jl4h+kEo+977GpmJEc2HlbxPw3z8ObpmNzdLiO8RI1M5L22Yyf4PvyNDmaqP7Jep7dT+nNu0n3ueV4l5EMqf721BZmxE01Lii/Dx58TqPdw+cJEcle74/E57c/qT37hXqKX/aXOfNojgPacI3n2KFN9wbi/ZQXpYHLUn6T6G3Cb2Jj00jttLdpDiG07w7lME7zmN+/RBmjRZialaZaZ9Vw9y0lVE6OkmXnl0mtKf04WO0b3lPEY9y3mMvpx/jKZX0TFaXl06tOHtNyfRp3vJZU9Vq6prPYA6O0fr2M2MS9aZriLqvDWIkN2nCNmVd87dWbyDjLA4ak/Wfc7VntibjNA47izOO+dCdp0iZM9p3GeUfM7ZdSt+zkUfu07MCS9S/SNJ9Y/k/ppfyU7NwLpVvUqP8WlTV/KrOhIVUgHIG3959epV/vrrLy5cuIBarWbgwIFkZWVp0qSlpfHJJ5/w008/8c8//xAcHMy8efM0n69bt45du3axbds2zp8/T1JSEvv37y/xO3///Xdq1qzJihUriIiI0KrcluXTTz/lhx9+4Pvvv+fcuXPEx8fzxx9/aKVZtGgR27Zt4+uvv+b27dvMmTOH8ePHc+bMmfL/YcpJIjPEqlkdok/7aC2PPnMTmzYv6FzHtlV9os/c1E5/2gfr5nWQSA01yxq9N5LMuCSC9pwuVz5cR3UmaE/lxwhgWcseMwcrgv8pyHdOZjahl+7h3Kp+ies5t6yntQ5A8BmfYutY1XFk6pWNTD73Gf03zcSyVkFXZEOjvBEGOaqCY1KdqyY3K4caJfyNH5dFLXvMHK0ILbRfcjOzibh4D8fWJcfn0Kqe1joAIad9SlxHYiCh7tD2yEzkRF3z1ZkGwMjCBHVuLqqktMeMpPxsXR1QOFhz72zBsZuTmY3fpbvUafXkf1cDqSGGUkOyC+0vgKyMTNzbNHzi7T4ua1cHLByseVgkvsBLd6lVgfgeGbLyNe6fuoHf+VsV3taTsHK1x8LBGv+z2udk0KV71CzlnHweSGSGKJrVIaZIuRpzxqfEctW6VX1izhRJf9obq+buWuVqYbXGdid8/wVy0lSVk/HHVN2P0eqkqq/15u5ODPDaTL/LG2izZTamtRyePBgdSjvnrFvrjs+qtY5z7pQ3ilLOOdex3Yko7ZwzkOA8vAOGpnISrpZ8jXxe5KKu1Fd1JMaQCvj6+vLXX39x/vx5OnbMuxu7a9cuXF1d2b9/Py+99BIAWVlZbNmyhbp187ohzpo1ixUrVmi2s3HjRhYsWMCIESMA2LRpE4cPHy7xe21sbDA0NMTCwgInJ6fHyvOGDRtYsGABo0bljafYsmULR48e1XyemprKZ599xsmTJ+nQIa/1zN3dnXPnzrF161a6dev2WN9XFrmNBQZSQzJilFrLVTFKjO0VutdxsEIVo12IZ8QoMZBJkdtYkBGdiE2bF3B7tTsnei8sVz5qDGiNTGFK0C/6qZCa2VsBkBarHWdarBJLF7sS1zO1t9K5jmmhv03kjYccm7OVBP8ITO0VtJ09nJd/X8rO3h+SkZhCgl8ESSExdPzgFU4u+J6sNBUt3xiImYMVZg5WlRKfaX586UXymh6rxLyM+HStY1pk39s0rMnwP5dhKJeRlZrB0Tc2kOgbrnObhnIZ7Ra8wsP9F8hKSX+CaMrHIj/m5CLHbnKMEuuaJcdcFlVqBgHXHtD37ZFEPgwjOTaRVkM7UbtFPWICIiuS5cdinr8PUorElxKThFUF4gPwGNIB5yZubBm2uELbqQjz/GO/aHypsUoUpRyzzwMjG0sMpIaodJSr8lLL1eLpDWRSjGwsUEUnan1m9WJdLBvVwnvuN5Wa98dhoedjtEYTN76uwmO0OqnKa3389Ydcnf01KX6RyO0VNJwznO4Hl3G82/tkJqRUPDgKzrlMXeecQ/nPucxSzjlF/jnnM6f4OWfRyJWOh1ZgIJeRk5rBtdc+I6WM+RuE6kFUSAXu3r2LVCqlXbt2mmW2trY0aNCAu3cLun2amppqKqMAzs7OREfndStRKpVERUXRtm3BmAFDQ0NatWpFbm7lzgmmVCqJiIjQVDQBpFIprVu31nTbvXPnDhkZGfTpo93FJDMzkxdffLHEbatUKlQq7Tt2WeocZBLdd/mKKXrjSqJjWeHkRQe653dbVqvVSM2MabN5BtfnfUdmfPm65bi92oOok95kVNIkOA2Gd6Tnmima939N/iQ/f0WzLSm2rKhin0skWguDCt2RjbsfSsS1h0w++ymNRnfhxndHyM3O4dBbX9B7/Ru8dfMbcrNzCD53m8CTXk8SGgD1RnSk69qC+I5M+iQ/szryWgZd8RXdv4l+Eezt9xFGlqa4D2xDj8+n8dfoVcUqpQZSQ3ptngkGBpxduL2c0ZRPq2GdeGX1G5r3W6es0x2ARMeyx/TTnM2M/XgaKy9/TU52DqG3Arj+53lqNq1Toe2WpvmwTgxdPbUgD1PyJszQdaoVO/8eg8LZhkFLJrJ94ppircD61HR4RwYVim/Pax/rTqjj+Htu6ShvSj02Szp/daxT69XuJN0NJvGGX8Xy+BiaD+vEsEL7cEf+Marr+lHRY3Twkolse8rH6H9CFVzrowpP7nMvhPhrvvS7+Dm1Xu7Kw60l3/x/EsVPoco751zH5p1zSh3nXMrDcM72/BCZwgynwW1p/uV0Lo5Y8dxXSsVzSMsmKqRCiRc8tVqtGdcJaM2OC48qIepiy8qz7dIYGBgUW69w1+HyeFQJPnToEC4uLlqfyeUlT4CzZs0ali9frrXsZbOmvGLuUer3qeKTyc3OwbjIHUS5nYKMIi1nmnWiEzEu0rJnbGdJblY2mQkpWDaoiVktBzrsKOgWLTHI+/sOD/2Jvzu9R2pQwTgTk5p2OHRtysUpn5ea18fh//d1IgtdNAzleUWGmb2CtEJ3PU1sLYu1gBaWFpOIWZG7x6a2lqTFJpW4Tna6irj7IVjVcdQsi74ZyO4BH2FkYYKhTEp6fDKv/LmMKJ+Axw0NgKBj19lbOL78bsEmuuKLKT2+oq2hJraWpBeJLzcrh6T8GYFjfQKwb+6Ox9T+nP3wB00aA6khvbfMxrKWPQdeXlPpraO3jl8jyOuh5r3UKO+8tnCwIikmUbPcwk5Bcin7tDzigqPY+MoKjEzkGJubkBSTyKRN7xAXont8VGW4e/waIVrx5e1TCwcFKYXiM7OzJLUC8dXwcMfcXsH0A//TLDOUGlK7bUPaTezLshcmos6t/B8hD/6+TlihY/ZRfOb2ClIKHbNmthWL71mQGZ9EbnZOsZYZIztLVCWUHaroxGLp5YXK1cIMTYyoMbwj99f/VrkZL0NJx6i5g4LkQseouZ0lKZVwjM4ocoy6tW1I+4l9WaqnY7Q6exau9Y/kpKlQ3g3B3P3xepiVRnPO2es452LKf84ZlXDOGeSfcw9KOOfUWTmk5V8jld7+WLVwx+2N/tya//2ThvRMqK6PaqlMokIq0LhxY7Kzs7l06ZKmy25cXBwPHjygUaNG5dqGQqHA0dGRy5cv06VLFyBv4qEbN24Um6CoMCMjI3JycrSW2dvbExkZqVUh9vLy0vouZ2dnLl68SNeuXYG8CZWuXbtGy5YtNTHJ5XKCg4Mfq3vuggULmDtXe1bTI/XfKCF1AXVWDok+ATh08yD8SMEEIA7dmhLhqfsxCXHXfHHu21JrmUP3ZiR4B6DOziH5YTjHu7+v9XnjD15Gam6Mz+IdpIXHaX3mNqYbqlglkcdvlJnf8spKzUCZmqG1LDU6kVpdmhJzO2+2RgOZITXbNeTc2l9K3E7E9YfU6tKUG98XPJKgVlcPIkoZP2loJMW6ngthl4tPNZ+ZnFdJs3JzxKGZOxc+ebLJqrJSM8gqGl9UIjW7NiWuUHzO7RtyaXXJ8UVfe0jNrk25+V1BfDW7eRBVxtgXiUSiqQRDQWVU4ebIgZdXo0qsnG5YhalSM1AViVkZnUCDzh6E3Q4EwFBmSN12jTiwdnelfGdmuorMdBUmlmY07NqMv9ZUznZ1fldqBvFF4kuOTqBuZw8i8vepocwQt3aNOLZ2zxN/j9/5W3zZV/v8HPnxNGL9wvlnywG9/dDPTM0gU0d8dTp7EFnomK3driEn1v6slzw8LeqsHJQ+Adh3a0ZkoXLVvpsHkSWUqwnXfHEsUq7ad29Gorc/6mzta43z0PYYGEkJ3Xeu8jNfipKO0Xo6jtGjFTxGvyhyjI76eBoxej5Gq7Nn4Vr/iIGRFMv6NYi7dK+CURUofM5FFYrPrqsHUUd1x5d41RcHHeecUsc5VyP/nAvbW85zTiLBwEhWdjrhuScqpAL169dn2LBhvPHGG2zduhULCws+/PBDXFxcGDZsWLm3M3v2bNasWUO9evVo2LAhGzduJCEhoViraWFubm78888/jBkzBrlcjp2dHd27dycmJob169czevRoPD09OXLkCJaWlpr13nnnHdauXUv9+vVp1KgRn332GYmJiZrPLSwsmDdvHnPmzCE3N5fOnTuTlJTEv//+i7m5OZMmTdKZH7lcXqwFtbzddX23HqbNxhkkePsTf9UXt/E9MXWxwz//WWNNFr6CsbMN12Z/DUDAjhPUndIXj2XjCdx1EpvW9XF7tTuXp28EIFeVRdI97edAZiXlzZBYdDkSCbXHdCXo17Ooc/R7L+7G9560mTmUxIAoEgMiaTNrKFkZmdzf/68mTd/Pp5ESmcC/634FwOuHo4z+bRGtpg/G/9g13Pu2wrVzE34btVKzTuePXiXg+A2Sw+MwsbWk7dvDMDI34e7es5o09Qa1JT0umeTwWOwauNJt2QT8j14l+GzlTdZx83tPXpw1FGVAFMqASF6cPZTs9EweFoqvx4ZppEYmcHntr/nrHGXovkU0nzGYoKPXqN2vFS6dm/DXyIL42n7wMsGnvEkJj8PI3Ji6Qzvg3KERh/OfwSYxNKDP1rex83DjyKRPkRgaYJJ/l1qVmEJulvaFvTKd+eEIfWYOJzYwkpiACPrMHEFWuoprf57XpBn36QyUUfEcXJ9XyTGUGeJUP286fqnMEIWjDS6Na6NKzSA2KO8Od8OuzUAiIdovHHs3J4YuHEe0fwSXfjutt1h0+fcHT7rNHEZcYCRxAZF0mzmMrPRMvP8s2KejPp1OUlQ8f6//RROffX58hjIplo42ODWunVeZCIoiMzWD6AdFzs90FWmJKcWW69vl7z3pPHMo8YGRxAdE0nnWMLIyMrlVKL5hn71FcmQCJ/PjMygcn5EUCydrHPPjS8jffzJTOTZuBa0vVq72ODauTXpiCkkl/EiubP5bD/HixpkkevuTcPUBtcf3wsTFTvNc0YYLx2DsbI1XfrkauOM4blP60njZeIJ3ncS69QvUerUH1/PL1cJqvdqDSM+rZFXS+LuKOF/oGI0NiKS7jmN0dP4xeqzQMepQ5Bh1zj8HSzpGM6voGC2vtLR0gkMLhjCEhUdx74EfCksLnJ0qdwKfJ1VV1/qmS8cSeew6aWFxyG0taThnBFILE4J+PUtlCthyiBab8s65xKsPcJ3QC5OadgT/mHfONfhoDMZO1njnxxe04zi1p/al0fLxhOw8iVXrF3Ad24MbbxU/51zH9iCqhHOuwcJXiD7hRUZ4HFJzE2oM74Btx8ZcHrO2UuOrCtV1IqLKJCqkAgDbtm3jnXfeYfDgwWRmZtK1a1cOHz5crJtuaT744AMiIyOZOHEihoaGvPnmm/Tr1w9Dw5IrdCtWrGDatGnUrVsXlUqFWq2mUaNGfPXVV6xevZqVK1cyatQo5s2bxzffFAyAf++994iIiGDy5MkYGBgwZcoURowYgVJZ0GVm5cqVODg4sGbNGvz9/bGysqJly5YsXFi+CYIeV9ifF5Fbm9Nw7si8h2XfC+X8uPWkh8YCYOxohWmh53ilBcfw77j1NFs+AffX+pARlYD3oh8JP3Tlsb/boWtTTGval2sm3oq69vVBpMZG9PjfZOSWpkR6+bF/3DqtlkaLGnZad98jrvlyZNYmOsx7iQ7vjUYZFMWRmZuI8iroemjubEP/TTMxsbYgPT6JyOsP+XX4UpLDCn74mjlY0XXxOEztFKRGJ3J33zkuf6k9u3JFeX+VF1/n/01GrjAl2suPQ0XiM3fRji/qmi/HZ26izfyXaDNvNElBUZyYsYnoQl0rTewt6fnFW5g6WJGZnEbc3RAOj19PWH5l2szZBrd+rQB46e/VWnn666X/EVHCc+kqw4ktfyEzNmL0yimYKswI8nrI1xNWa7WkWrvYaXWlVzja8P7hdZr3vaYNode0IfhevMOmMXmTnRlbmDLk/VexcrIhVZmC95HLHPrkZ3Kz9Ve51uXslgPIjI0YuvI1jBVmhHr5sX3CGq2WRisXW9Tqgps5Fo7WzDq8RvO+y7TBdJk2mICLd/h+zKqnmv+y/Lsl75gdsGoyJpZmhHn5sWv8Wq34LGvYah2zFo7WvHmk4DjrOG0wHacNJvDCHX4ak9fFs0Yzdyb+skiTpu+SCQB4//YPf83bqu+wAAj/8yIyawtemDsSuYMVyfdCuDRunVa5alJo8qb04Bguj1tPk+UTcHutL6qoBG4t+pGIQ5e1tmvm7oRt+4ZceFn7XKsquo7RbUWOUUU5j1H/Z/AYLa9b93yZMrvg2ZrrN+Zd94cN6M3/Fr1XVdnSUlXXehNnW9p8PRu5jQWquCTirz3k9KClmu+tLBF/XsTI2oL6c0cid7Qi5V4IV8YWnHNyh+Ln3JWx62m8YgK188+52x/9SKSOc86mfUMuvaT7nJPbK2ixaSZyRyuyk9NIvhPM5TFriS0yQ//zSIwhLZtEXW1mPRCeNbm5uTRq1IiXX36ZlStXlr3CM+p3p7FVnQW9CjGq3k9/Mv4PDN64Iy35GYTVgcV/4AllcnXZE2c9z1qoqveJeNG4eu8/gGVXn89KbnkdaLqo7ETPMfl/4Of+oKgn7+KuT6NrD63U7e0N+qtSt/csEC2kQqUJCgri2LFjdOvWDZVKxaZNmwgICGDs2OpdoRMEQRAEQRAEXar37bjKISqkQqUxMDBg+/btzJs3D7VaTdOmTTl+/Hi5J0YSBEEQBEEQhOpEdEYtm6iQCpXG1dWV8+fPl51QEARBEARBEAQBUSEVBEEQBEEQBEHQCzHLbtlEhVQQBEEQBEEQBEEPxBjSslX/qQsFQRAEQRAEQRCEZ5JoIRUEQRAEQRAEQdAD8RzSsokKqSAIgiAIgiAIgh6IMaRlE112BUEQBEEQBEEQhCohWkgFQRAEQRAEQRD0QDyHtGyiQioIgiAIgiAIgqAHYpbdsokuu4IgCIIgCIIgCEKVEC2kglCGA8aqqs6CXg1UGVd1FvSqtUN0VWdB73rnVO97i+lpsqrOgt4lpFXv8/CmrHrHl01OVWdB7w40XVTVWdCrIbdWVXUW9CprW/WO71kmZtktm6iQCoIgCIIgCIIg6IGYZbds1fu2uiAIgiAIgiAIgvDMEi2kgiAIgiAIgiAIeiBm2S2bqJAKgiAIgiAIgiDogeiyWzbRZVcQBEEQBEEQBOE/LCEhgQkTJqBQKFAoFEyYMIHExMRyrz9t2jQkEgkbNmx47O8WFVJBEARBEARBEAQ9UFfyP30ZO3YsXl5eeHp64unpiZeXFxMmTCjXuvv37+fSpUvUqFHjib5bdNkVBEEQBEEQBEHQg9znYAzp3bt38fT05OLFi7Rr1w6Ab7/9lg4dOnD//n0aNGhQ4rphYWHMmjWLo0ePMmjQoCf6flEhFQRBEARBEARBeA6oVCpUKpXWMrlcjlwuf+JtXrhwAYVCoamMArRv3x6FQsG///5bYoU0NzeXCRMmMH/+fJo0afLE3y+67AqCIAiCIAiCIOiBupJfa9as0YzzfPRas2ZNhfIYGRmJg4NDseUODg5ERkaWuN66deuQSqW8/fbbFfp+0UIqCIIgCIIgCIKgB5U9y+6CBQuYO3eu1rKSWkeXLVvG8uXLS93elStXAJBIJMU+U6vVOpcDXLt2jS+++ILr16+XmKa8RIVUEARBEARBEAThOfA43XNnzZrFmDFjSk3j5uaGj48PUVFRxT6LiYnB0dFR53pnz54lOjqaWrVqaZbl5OTw3nvvsWHDBgIDA8uVRxAVUqGShISEsGzZMo4cOUJsbCzOzs4MHz6cJUuWYGtrW65tBAYGUqdOHW7cuEGLFi0qPY8SiYQ//viD4cOHV/q2yzLs3Zfp9mofzBRm+Hv58tPi7wj3DSkxfdcxvek0shsuDfJO8sCb/uz7eBcB3g+1tjn83Ve01lPGJPBum9f1EwRQd1JvGswYhLGDFUkPwvBa8hOxl+6XmN6uQ0NaLBuP5QsupEclcv+rg/jvOKH53GVgaxq+PQxzN0cMZIak+Edxf+thgvee06Rp/N5ImswbpbXdjOhEDjSfWfkB6mDx8hAUk1/C0M6WLL9A4tZ/jerGLZ1pTXt1xvKlwRg1qIvESEamXxCJW34i/d+rmjRO332CSZvmxdZN++cSUbMX6S2Okli+Mhir117C0N6GrIdBxK7bQsZ13fGZ9e6E5SuDkTdwz4vvYRDxX+0k/d9rmjQWw/rg8L95xdb1bzkYdWaW3uIojc34gdi9MRKpgw2qB8FErPqWtCu3daa17NcBm3EDMW6UF6PKN5joL3aTcva6Jo28fi0c5ozDpGk9jGo6ErHyG+K2/fW0winGeXI/as4YipGDNan3Q/Bfsp2kS3d1ppU5WOG+bBLmzdwxcXcm/LvD+C/ZrpXGdmA7XN8ZiYmbExKZIen+EYRtOUD03n+eQjR52swZSeNxPZArzIi64cc/i7aT8CCs1HXcB7Sh7bzRKGo7oAyK5tLHvxHgeVVn2pYzh9D+w1fw/s6T88t36kzTbc0UmozvybllP+Hz/dEKx1SW3u+Oot2rvTBRmBHs9ZA/F28jyje0xPSO9WvSZ+5oXDzcsalpz4EVOzj3w5Fi6SwdrRnw4VgadG+OzNiI2IAI9r7/DWG3AvQWi/vk3tSfMTjvWnE/DJ8lO4gr41rhsWwClg1cyIhK5MHmAwQUulYUVnNYB9punU34katcfO0zzfJG80bRSMe14nCzGZUTVCW46nWTbbv3cufeQ2Li4vlizWJ6de1Y1dkqF2mLHkjb9EdiboU6NozMk3vIDfPVmdbAtQHGYz4otjz9+4Wo4/O7fxoYIm03EGnTTkjMrVHHR5J55jdyA3Vff55nVfkcUjs7O+zs7MpM16FDB5RKJZcvX6Zt27YAXLp0CaVSSceOuo/RCRMm0Lt3b61l/fr1Y8KECbz22muPlU9RIRUqzN/fnw4dOvDCCy+wZ88e6tSpw+3bt5k/fz5Hjhzh4sWL2NjYVHU2q8zAt4bTb+oQvp+3iciAcIbMHs28nUtY2HM2GakZOtdp2L4JF/86x8Pr98lSZTFw2jDm/bSEj/q8S2JUvCZd6P1gPh5f0BVDnZOrtzhqDm1PixUTuL5gG7FXHuA+oSdddr2PZ7f3SQ+LK5be1NWeLjvn47/rFJdmfYVdmxdoueY1VHFJhB3K6x6SmZDK3S/+JPlhOLmZ2Tj3eZE2n7+JKlZJ1Ombmm0p74Vw5uWC8RHqXP3FWZhZv27Yvj+d2P9tROV1G4vRg3D6ajWhI6aSExlTLL1xSw/SL14nfuMP5CanYjGsH45friB8/Gwy7/kBED13ORJZQdFrYGWJy69bSf376f3Yf8SsfzfsPnyLmFWbyLhxG8uXBuG8ZRUhQ98gW1d8rTxI//c68V9sIzcpBYsR/XDevJzQV9/RxAeQk5xKyOCpWutWVWXUclAXnBa9QcSSr0m7dgfrsQOo/cMyHvabQVZ48RhN2zYl5ZwXUZ/sICcpFevRvan17WL8R75Hxh1/AAxM5GQGR5J0+DxOi/R3A6g87IZ1xH3FZB5++B1JV+7hPKEPTXcv5FrXOajCYoulN5DLyIpLIuSL33F5c7DObWYnphCyYR9pD8NQZ2Zj06cVL2yYSWasksTT3voOiRenD6b5GwM4OXcriQGRtHp7GEN3f8jubvPJKqHMdGxZj75fzeLyJ3vx97yKe//W9P1qFn+MXEm0l59WWofm7jQe24PYO0El5qFOv1Y4vliXlMj4EtNUpm5vDaHL1IH8Om8LsQER9Jo9gtd3LuTjnnPJLCFmmYkR8cHR3Dx8icGLdT+awcTSjOn7luN/4TY/TF5HapwSm1qOpCel6i0Wl2HtabZiIl4f/kDclQfUmdCLTrs/4O+u83VfK2rZ03HX+wTuPMXVWZuxbfMCLdZOQRWXRHj+tUITT007PJaOJfaC7hsuynshnHtpteb907pWlFd6egYN6rkzfGBf5ny0qqqzU26GDdog6/kqmX//RG7YQ6TNuyMfPYeMHxahTi75HEn/bgHqzPSCBWnJmv/KOo/AsHEHMo9uRx0fiWGdJsiHzyJj92rU0cH6DOepUz8Hs+w2atSI/v3788Ybb7B161YA3nzzTQYPHqw1oVHDhg1Zs2YNI0aMwNbWtlijk0wmw8nJqdRZeXURkxoJFTZz5kyMjIw4duwY3bp1o1atWgwYMIDjx48TFhbGRx99BOS1UO7fv19rXSsrK7Zv3w5AnTp1AHjxxReRSCR0794dgMmTJzN8+HCWL1+Og4MDlpaWTJs2jczMTM123Nzcij2It0WLFixbtkzzOcCIESOQSCSa909DnymDObh5H9eOXiLsQQjfvbcRuYmc9sO6lLjON+9+wamdRwm5E0ikXxjbPtyCRCKhcScPrXS5OTkkxSRqXsnxSXqL44VpAwjYc5qA3adJ9g3He8lO0sLjqDupt870dSf2Ii0sDu8lO0n2DSdg92kCfj5Dg7cKpgSPuXCX8CNXSfYNJzUomoffHUV5Nxi7ttoFmTo7F1WMUvPKjEsu+nV6YTlhFMl/eJLyxxGyAoKJ//hrsiNjsHx5iM708R9/jXL7r2TefkB2cBgJG38gKzgM024dNGlyk5LJiUvQvEzat0SdkVElFVKriSNJ+v0oyfs8yfIPIW7dlrz4xuiuqMSt20Litt9Q3XpAVnA48V9sIysoHLPu7bUTqtVaMebEJTyFaHSzmzqchN/+JuHXY6j8Qolc+S1ZEbHYjBuoM33kym+J/WYf6T6+ZAaGE/XJDjIDw7Ho1VaTJt3Hl6i121Ae/KfKKtqPuEwbQtSek0TtPkG6bxj+S7ajCovDeVJfnelVITH4L95G9G9nyE5O05lG+e9t4o5cJt03jIygKMK/O0zqnSAUbRvpMxSNZlP7c23jn/h7XiX+fign5mxFamxE/eEltyQ1f70/IWdvcX3zARL9Iri++QBh5+/Q/PX+WumkpnJ6fzmd0x98j0qpO34zJ2u6rJzE329/RW5WTqXGVpLOUwZwcvN+bh+9QtSDUH5572tkJka8OKxTieuE+vhzeM1uvA9cIDszW2eabtOHoAyP47f5Wwn19iMhNBa/f28THxytr1CoP20ggXtOE5h/rfBZ8hNpYXG4l3CtqDOxF2mhcfgs+Ylk33ACd58mcM9p6k8vUg4ZSGizeSZ3Pt5Hagn5V2fnVMm1ory6dGjD229Ook/3kvfrs0jauh/ZN8+Sc/Ms6vgIsk7tQZ0cj7RFj1LXU6clQWqhV6GKmWGTjmRfOkRuwE3UyhiyvU6TE3gLWZt++g5HKMGuXbvw8PCgb9++9O3bl2bNmvHTTz9ppbl//z5KpbLSv1tUSIUKiY+P5+jRo8yYMQMTExOtz5ycnBg3bhy//PJLue4OXb58GYDjx48TERHB77//rvnsxIkT3L17l1OnTrFnzx7++OOPMgdpF/ZowPa2bduIiIjQvNc3e1dHrBysuXW2oFUhOzOb+5duU69V+e8eyU2MMJQZkpqYorXc0c2Zzy59y/qzX/HWxjnYu+ru519REpkh1s3qEHnmptbyqDM3sWtdX+c6tq3rE1UkfeRpH6yb10EiNdS5jkPnJljUdSbm4j2t5ebujgy+sYmBlz6n3dezMKtlX4FoykkqRd7oBdIvXNNanH7hGvLm5ZzaXCLBwNSUHGXJP4osRgwgxfM06nTdrSB6I5Uib1xfq7stQNq/1zBu3rh825BIkJiZFIvPwNSEWsd2UPv4Tpw2r8CoYd3KyvVjkcikmDStR8rZG1rLU87ewLRlw3JuRIKBuQk5Rc69Z4FEJsWimTsJRVotE854Y9nm8e5Ol8aqswcm9WqgvHin0rZZEsta9pg5WhHyT0HZkZuZTfilezi10l3WQF4LaeF1AILP+BRbp+uqyQSd9CL0nO4u20gk9NrwFl5bDpXZRbiy2Lg6YOlgje/ZgvznZGbjf+kutVu9UKFtN+7ditCb/ozb/A6Lr27h7UNraDumZ0WzXCKJzBCrZnWIPu2jtTz6zE1s2uiOxbZVfaKLXCuidVwrGr03ksy4JIL2nC7x+83dnRjgtZl+lzfQZstsTGsVnzVUeEwGhhg41SY3UPucyQm8jYFLvVJXNZ64DJPpnyF/eR4GrtplrsRQijq7yA297CwMXEo+z59Xuagr9aUvNjY27Ny5k6SkJJKSkti5cydWVlZaadRqNZMnTy5xG4GBgbz77ruP/d2iy65QIb6+vqjVaho10n3nvFGjRiQkJBATU7xrXFH29nmVDFtbW5ycnLQ+MzIy4ocffsDU1JQmTZqwYsUK5s+fz8qVKzEwKPu+yqNtW1lZFdu2PinsrQBIiknUWq6MUWJXs/yVqtEfjCchMp7b5wsu8v5evnw7dyNRAeFY2lkxZPYoPvr9f3zU591iFdeKkttYYCA1RBWjfVcsI0aJsb1C5zrG9goyiqRXxSgxkEmR21iQEZ0IgNTChCE3NmFgJEWdk8v1BduJ/qdgDEn8DT8uv72FZL9IjO0tafTucHoeWMbR7h+QmaC/SoKhtQKJ1LBY615OXAKGdtbl2oZi4mgkJsakHjuj83Ojpg0wql+HmGWfVji/j8vQ2hKJ1JDsuESt5TlxieWOz2ryKAxMjEk9WhBfZkAI0Ys+IdM3EAMzUxTjh+Py02eEjppOVnB4ZYZQJk2MscX3odS+Zbm2Yff6CAxMjFEePquPLFaIzMYCidSQzCLnWVaMEll+2fOkDC1Maee1FYmRDHJyebjgOxL/8Sl7xQoyzc93Wqx2TGkxSixqljwOytTeivQi66THKjEtVD7VG9oeew839g5eUuJ2Ws4YjDonF58f9D9m9BGL/DwmF9mPKTFKrEuJuTxsajnQfnxvzn53mFNf/Ylr87oMXTaJ7Mwsrv9e+cf0o2uFrrK/pGuF3MEKVYz2sZVR5Fph0+YF3F7tzoneC0v87vjrD7k6+2tS/CKR2ytoOGc43Q8u43i39/V6rajuJCYWSAwMUadq71N1ahISM937VJ2iRHV0O7mRgUikMgwbd0D+yjxUP68nN/QBADkBt5C17ktuyH3UiTEY1G6EYb0WIKl+bWXqKhxD+rwQFVJBrx61jFZ0OujmzZtjamqqed+hQwdSUlIICQmhdu3aFdp2YboeNpyjzsFQortFr6j2w7owafU0zfsNU/LGshRtIZZIyj+mYMC0YbQb2pl1Y5aSrSq4m3jzdKFWn/vBPLx+n/X/bKbTqB4c+/5Aubb9uIrlWSIpvZgt+mH+cVB4O9kpGRzrvRCpmTGOnZvQfNk4UoOiickfIxR5sqD1J+kexF19yMCLn1H75S74bi0+gUel0xFzsWU6mPXvgdX0CUS9s5Tc+ESdaSxG9CfTN4DMWyVP9qF3OuMrezXzAd2xnj6ByLeXkRNf8ENF5XMPlU9BC3fGjdvU/G0zluOGEbfm68rK9eMpFk/59qFiSFcc3hlL0LSV5MRVfhelSlNsH1KufVianJR0rveaj6GZMVZdPHBfNomMoCiU/5bQsviE6g/vSPe1UzTvD03+JO8/xQ7Lso/L4rtUoilrzJ1t6LxsAgfGrSNHpbubtb2HG82m9OPXgfqdXKzFsE6MXF0w9njblPV5/yl2nZCU5zAtlURiQNhNf45+/AsA4bcDcaxfk/bje+ulQqpRrOzXsaxwcl3lUP5yqZkxbTbP4Pq878iML7m3SVShawX3Qoi/5ku/i59T6+WuPNx6+PHyL5StlGuhOiGSnIS8yYvUQG64HwYWNkjb9CMzv0KaeXIPRv0mYTx1NaBGnRhD9q3zSJs+X92ZhcohKqRChdSrVw+JRMKdO3d0zl577949rK2tsbOzy7+4ahdeWVkVG3/1qKJrYGBQKdtes2ZNsa7AzRUNedGqfF0YvY5fwd+rYNY5qZEMAIWDNcpCraSWdgqSYhMpS/83hjJ45ig+Hrec0HslT8ABkJmuIvReMI51nMuV18ehik8mNzsHYwcrreXGdpbFWk0fyYhRYuygffdUbmdJbla29t1qtZrUwLypxpW3g7CoX4OGbw/VVEiLyklXobwbgkUd/bZ05yQoUWfnYGinPSGXoY0VOUVaFYsy69cNu2VziZ6/koxLN3SmkRjLMe/Xg4SvfqysLD+WnIQk1Nk5SO2sKXwLxtBGUeaYT7P+3bBfMYeo9/5H+kXd8Wmo1ahuPcColkvFM/2YNDHaa7f4GtpakV3G+Wc5qAsua98meNZaUs/rfyKfJ5EVn4w6OwejIuelzE5BVjnKl1Kp1WQE5v2gTL0diGl9F1xnj6j0Cmng39f5pdCkQ4ZGeT9LTO0VpOX3ogAwsbMkrYSyBiAtJlGrNfTROumxeePq7ZvVwdRewUuHV2o+N5AaUqNdAzwm92Fr3ck4t22AiZ0lEy9+oZWm4+JxNJvan50d51Qo1kfuHL9GiFfBjOmPrhMWDlYkF7pOmNlZkhJbsRshydEJxWbqjfYLo+mAtiWsUTEF14qiZb+CjBJiUUUn6ry2PLpWWDaoiVktBzrsKJi9W2KQd+0fHvoTf3d6j9Sg4mNKc9LyrhXm7k+vV1R1pE5PRp2bU6w1VGJqkTdGtJxyIvyRNi4030B6Mpn7N4GhFImJOeqURGRdR6NWFp+M7Xn3PExqVNVEhVSoEFtbW/r06cNXX33FnDlztMaRRkZGsmvXLiZOnIhEIsHe3p6IiAjN576+vqSlFUwqYWRkBOQ9w6gob29v0tPTNdu/ePEi5ubm1KxZE6DYtpOSkggI0J7SXiaT6dx2YboeNjzLY2Kp6xSWkZpBRmqk1rLE6ASadG5G8O28/BjKpDRo14Tf1v6kaxMa/d8cxpBZo/h00koCb/qVmhZAaiTFuV5NHlzRXZGrCHVWDgk+ATh2bUr4kYLHKDh29SDs6DWd68Rd9aVGX+1ukU7dPEjwDkCdXfJ+kEgkGBiVXDQZGEmxqO9CTCmPEKgU2dmo7j7ApH1L0k6e1yw2ad+StNP/lriaWf8e2C1/j5gPV5N+9nLJ6fp2AyMZKYeOV2q2yy07G9UdX0w6tCT1REE8ph1aknrqQomrmQ/ojv3KuUS9v4a0f0qOrzCjhu5k+gZWNMePTZ2VTfqth5h3bkHysYKYzDu3IPn4pRLXUwzpisu6dwh552NSTul+bMizQJ2VTbKPP1bdmhF3pGBfWHdrRpxnJY+Tl0iQyGWVu00gKzWj2My5qVGJ1OzSlNjbeTfhDGSG1GjXkAtrfilxO1HXH+LapSk+33lqlrl29SDyWt4NwtBzt/m594da6/T89E0SHoZz4+uDqHPV3N93vtjY0sE73+fBvvPc+7XyJh3LTM0grkjMSdEJ1O/sQfjtQAAMZYa4t2vEkbV7KvRdgdceYO9eQ2uZXR1nEnXMwFwZ1Fk5JPoE4NDNQ+ta4dCtKRGeJVwrrvniXORa4dC9meZakfwwnOPd39f6vPEHLyM1N8Zn8Q7SwovP3At51wrL+jWIu3RP5+dCOeXmkBsZhEHtxuT4Fjz+yrB2E3IelnFDshADh1qoU3TclMjJRp2SCAaGGL7Qipz7T2eOj6epKh/78rwQFVKhwjZt2kTHjh3p168fq1at0nrsi4uLC//73/8A6NmzJ5s2baJ9+/bk5ubywQcfIJMV/MBxcHDAxMQET09PatasibGxMQpF3h25zMxMpk6dyqJFiwgKCmLp0qXMmjVLM360Z8+ebN++nSFDhmBtbc3ixYsxNNTuZuvm5saJEyfo1KkTcrkca+vi4+R0PWy4vN11S/L3DwcZPHMUUYERRAVEMHjmKFTpKi7+WdBd6vVPZ5MYFc/e9buAvG66I+a+ytZ3NhAbGoNl/rgqVWoGqrS8HzKvLJyI14mrxIXFYmmnYMis0ZiYm3B+3+kK5bckD7Yeod3G6SR4BxB3zRf38T0xdbHVPFe06cJXMHGy5srbWwDw23GCelP60HzZOPx3ncK2VX3qvNqdizM2abbZcPZQ4r39SQ2MwsBIilOvFtR+qTPXP9ymSdNsyVjC/75OWmgcxnZ5Y0hlFiYE/ab/MX1JP+3D/n8foLrzAJX3XSxGDUTq7EDybwcBsH57CoYOdsQuyutyZ9a/B/ar3idu/VeofO5iaJt3jOWqVKhTtGf0tBjRn7RT58ktZcIjfUvc8TuOa+ajuv2ADO+7WI7Oiy/pl0MA2Lz7GlIHO6IXfgzkVUYdVs8ndu3XqLzvaeJTq1Tk5sdnPX0cGd73yAoO04whlTeoS+yqzVUSY+z3+6n56VzSbz4k/fpdrF/tj6yGPfG78rrwOc6fhNTRlrB5ec8zVAzpSs1P5hKx8hvSb9xDamcFQK4qk9z8WWklMinyeq6a/0sdbTFuVIfctAwygyKKZ0KPwrYeoMHG2aR4+5N09T7O4/sgd7EjYscxANwWjsXI2ZYHszdq1jFr4gaAoZkxMlsFZk3cUGdlk/YgryWt5uwRpHj7kREYicRIik2vlji81I2HH3z7VGLy+d6TVrOGogyMQhkQSctZQ8nOyMR3f8GNk16fTyM1MoGL637NX+cow/cu4sXpgwk4do06fVtRs3MT/hiZ1yKalZpB/H3tlsKsNBUZCSma5arEFFRFxt/nZuWQFpNIor9+9+u5H47QY+YwYgMjiA2IpMfM4WSlZ3Ljz4KbYS9/Op2kqAQ81/8M5FVaHern3ZSVyqRYOlrj3Lh2XoU3KK/XybnvDzNj33J6zBiGz6GLuDavS7tXe7JvwXd6i8V362HabJxBgrc/8Vd9cRvfE1MXO821osnCVzB2tuHa7Lwu/AE7TlB3Sl88lo0ncNdJbFrXx+3V7lyennfM5qqySLpXZN/lP7am8PKmS8cSeew6aWFxyG0taThnBFILE4J+fXbGf6elpRMcWjCWPiw8insP/FBYWuDs9OxOwJR99ShGg94gNzKQ3HA/pM27IbG0Idv7NACyLqOQWFiTeTjvuJK26oNaGUtubBgYSpE27oC0QWtU+wuu/wbO7kjMrciNDkFiboWs0zCQGJB1+SkMxRGeOaJCKlRY/fr1uXr1KsuWLeOVV14hLi4OJycnhg8fztKlSzXPIP3000957bXX6Nq1KzVq1OCLL77g2rWCO6ZSqZQvv/ySFStWsGTJErp06cLp06cB6NWrF/Xr16dr166oVCrGjBmjeaQL5LVs+vv7M3jwYBQKBStXrizWQvrpp58yd+5cvv32W1xcXAgMDNT3nwaAw1v2IzM2YsLKNzFTmOHn5cunE1ZoPYPU1sVOq0tHzwn9kcllzNoyX2tb+zf8wp8b8n6AWTvbMu3LOVhYW5Acn4TfDV9WjVhAXFjZE0g9idC/LiK3Nqfx3BH5DzsP5ez4j0kLzbvTbuJghalLwfOo0kJiODv+Y1osH0/dyX3IiErgxuIdmmeQAhiaymm55jVMnW3Iycgk6WE4l2Z9TehfFzVpTJxtaP/VLOQ2Fqjikoi7/pATg5dqvlefUo+ewUBhidWb45Ha25D5MJComR+RHZHXPczQzhZpoR8RFqMHIZFJsfvobfjobc3y5D+PEbvkY817aW0XjFt6EDGt+IPDn6ZUzzPEKiywfmtcXny+QURMX1QoPhukzgWTb1m+PBCJTIr94tnYL56tWZ60/xgxi/ImZjKwMMd+2TtI7azJTU5Dde8hYZPnoaqicbJJh84SaW2Bw+wxSO1tUD0IImjKMs0zSKX21hjVKIjR5tUBSGRSaqyYQY0VMzTLE/YeJ+z9DXnrONhQ71BBBc/+zVHYvzmK1Is3CRi74KnE9Ujsn/8is7ag1tzRGDlYk3ovmFvjVqPKPz+MHK2Ru2hPjNPyxCea/1s0r4vDqC5khERzpU1evIamcuqtfQMjZxtyMzJJfxjO/VlfEvtnyT0DKtONrw8iNTai66rJyBWmRHn5cWDcOq2WVPMiZWbkNV+OzdxEu/kv0XbeaJKCovh7xqZizyB9Vp3ZcgCZsRHDV07BRGFGiJcf301YrfUMUqsiMVs6WvPu4bWa992mDaHbtCH4XbzDN2PyKuKhPv7smPYZ/d8fQ693RpIQEsOBFT/hVaiiW9nC/sy7VjScOzLvWnEvlPPj1pOef0waOxa5VgTH8O+49TRbPgH31/KuFd6Lfiz2DNKymDjb0ubr2ZprRfy1h5wetFTzvc+CW/d8mTK7oNxfv/EbAIYN6M3/Fr1XVdkqU879K2SZmCPrOBSJmQJ1bBiqfRtQJ+W1TkvMFUgsCg1vMTRE1v1lJObWkJ1Jblw4GXs/Jzeg0GzKhlJknUcisbKHzAxyAm6Seeg7UKVT3Yguu2WTqMVfSXjGTZ48mcTExGLPMH1aXnMbVSXf+7QMVBlXdRb0qrWD/p6396zIzal+sxIWlp5W+V1FnzUJadX7PLwpq97xBRk+neeVVqV2GRWbnPBZN+TWqqrOgl5lbave8QGYzv+hqrOgU3Onkp+h/CS8I5/OzcGnqXr/ihEEQRAEQRAEQRCeWaLLriAIgiAIgiAIgh6I55CWTVRIhWfe9u3bqzoLgiAIgiAIgvDYcsXoyDKJLruCIAiCIAiCIAhClRAtpIIgCIIgCIIgCHoguuyWTVRIBUEQBEEQBEEQ9EB02S2b6LIrCIIgCIIgCIIgVAnRQioIgiAIgiAIgqAHostu2USFVBAEQRAEQRAEQQ9El92yiS67giAIgiAIgiAIQpUQLaSCIAiCIAiCIAh6ILrslk1USAVBEARBEARBEPRAdNktm6iQCkIZLKr5aRIjlVR1FvRKmWhS1VnQOwNJ9b7YSar3IQpAlrp6j6Axya3qHOiXiUH13n8AcnX13olZ21ZVdRb0SvbaoqrOgiCUqHr/0hYEQRAEQRAEQagiostu2USFVBAEQRAEQRAEQQ/U1bx3QWWo/n1MBEEQBEEQBEEQhGeSaCEVBEEQBEEQBEHQg1zRZbdMokIqCIIgCIIgCIKgB2oxy26ZRJddQRAEQRAEQRAEoUqIFlJBEARBEARBEAQ9EF12yyYqpIIgCIIgCIIgCHoguuyWTXTZFQRBEARBEARBEKqEaCEVBEEQBEEQBEHQg1zRQlom0UIqPBXLli2jRYsWmveTJ09m+PDhT/U7BUEQBEEQBOFpUlfyv+pItJD+R02ePJkff/yx2HJfX1/q1atXBTn6bxnw7mg6vtoLE4U5QV6+/Lb4ByJ9Q0tM32FMT9qO7IpzA1cAQm4GcODjPQR7+z2tLGtpM2ckjcf1QK4wI+qGH/8s2k7Cg7BS13Ef0Ia280ajqO2AMiiaSx//RoDnVZ1pW84cQvsPX8H7O0/OL9+pWd7zszdp+FJXrbSR1x/y+7BlFY6pJLYTBuAwbSQye2syfIMJW/4dqVfu6EwrdbDGZdEUTJrWRV6nBrHbDhK24rsiiQxxnDEam9E9kTnaovIPI3ztjySfua63GB6X7fiB2E8bidTBmowHwYSv+Ja0EmK27NcB2/EDMGnsjsRIRoZvMFEbdpPyz42nnOvysxk/EPs38+JTPQgmfGXp8dmMH4BJo7z4VL7BRH3xbMXnMrkvtWYOxcjBitT7ofgu3o7y0j2daY0crKi3fCIWzdwxdXci9Lsj+C7WvhbUGN8Lp5e6YtYwr7xJ9vHHb/Uekm9UTXnzSIu5I2kwrgdGCjNibvhx8aPtJJZR7tQe2IaW80djUduB5KBorq37jeBC5Y5juwY0nT4IO486mDpZc2LK5wQfvabvUHTq/u5IWo3tibHCjLAbDzm0eDsxviXHZ1/fhR7vjaZG0zpYudrjufwnLv7gqZWm9fhetBnfG6ua9gBE+4Zy5os/eHjaW6+x1J7cB/eZg5E7WJFyP5Tbi3eQcOl+ieltOjSi8fLxmDeoiSoqAb9NBwnecVzzefvfF2PbqXGx9aL/vsGV8esBqDWpN7Un98HE1Q6AlPuh+H76OzEn9RvrI9IWPZC26Y/E3Ap1bBiZJ/eQG+arM62BawOMx3xQbHn69wtRx0fmJzJE2m4g0qadkJhbo46PJPPMb+QG3tJnGBV21esm23bv5c69h8TExfPFmsX06tqxqrMlPONEC+l/WP/+/YmIiNB61alTp6qzVe31fmsoPaYO4rcl2/h06EKSYpTM3PkRcjPjEtep374J1/76l42vruCzkYtJCI9lxk8foXC0foo5z/Pi9ME0f2MAZxf9yN7BS0iLSWTo7g+RlZJ/x5b16PvVLB78fo5f+i3kwe/n6PvVLBxa1C2W1qG5O43H9iD2TpDObQWd8mZby5ma16FJH1dabEVZDe6My5LXidr0K/cHvUvq5Tu4/7gUWQ07nekNjGRkxymJ2vQb6XcDdaZxnjce23H9CV36Dfd6zyR2lyd1vlmASRN3vcXxOBSDO+O85HWiN/2K78B3SL1ymzrblyGrYa8zvVm7JqSc8yLgteX4DnmX1As+uH23GONnJJ6iFIM647z4daI3/8rDQXnxuW0rJb62efEFTlnOw6HvknLBh9rfLsa48bMRn8OwDtRfOZnADb9zpfcHKC/dpfmehchdbHWmN5DLyIpLImjD76Tc1n2OWXVsTNQf57kxcjnXBi0iIyyOFr8swsjp6Zc3j3jMGEyTNwdwcdGPHBi0hPSYRPrt+RBpKeWOfat6dP96Fg/3nePPPgt5uO8cPbbMwu7FgnJHaion4U4wFxcVv0H7NHV6azAdXh/I4SXb+XbIYlJilEzctQCjUuKTmchJCI7m+LqfSY5O0JkmKSKe4+t+5pshi/hmyCIC/r3Nq9/Oxb6+i75CwXlYexqvnMjDDfs513sB8Zfu03bPhxiXcEya1LKnze73ib90n3O9F/Dwiz9p8r9JOA1qq0lzbcpnHG/6luZ1put8crNziDhwUZMmIyKee6v2cL7vR5zv+xFx527T+sd5mDeoqbdYHzFs0AZZz1fJuniQjB+XkRPqi3z0HCQWNqWul/7dAtK+elfzUidEaT6TdR6BtHl3Mo/vIuOHRWR7n0I+fBYSh1r6DqdC0tMzaFDPnYVzZ1R1Vp4ZarW6Ul/VkaiQ/ofJ5XKcnJy0XoaGhhw4cIBWrVphbGyMu7s7y5cvJzs7W7OeUqnkzTffxMHBAUtLS3r27Im3t/YdyLVr1+Lo6IiFhQVTp04lIyNDZx6WL1+u2c60adPIzMzUfObp6Unnzp2xsrLC1taWwYMH4+enfYc+NDSUMWPGYGNjg5mZGa1bt+bSpUs6vysgIIB69eoxffp0cnNzn/TPVmHdpgzk2OY/8Dl6mYgHIex6bzMyEzmthnUucZ0d727k3M5jhN0JItovnD0fbsVAIuGFTh5PMed5mk3tz7WNf+LveZX4+6GcmLMVqbER9YeXfAe0+ev9CTl7i+ubD5DoF8H1zQcIO3+H5q/310onNZXT+8vpnP7ge1TKNJ3bysnMIj1GqXmpElMrNb7C7F8fRvwvx4n/+W9UD0MJW/EdWRGx2I0fqDN9Zmg0Ycu/I+H3U+Qm6c6XzcjuRG/+jeRT18gMiSJu5xGSz9zA/o3heovjcdi/PpyEX/8m/pdjqPxCiciP2Xb8AJ3pI1Z8R8zW30n38SUzMILIj38iMzACy15tdaavanb58SU8im9lXnw240qIb+V3xBaKL+qTZys+17cGE777JBG7TpLmG4bv4h9RhcXiMrmvzvQZITH4LtpO5G//kJ2k+xy7M2MjYduPkXI7iLSH4dybuwWJgQSbLk+/vHmk8ev98fnyT4KOXCXxfihn392KoYkRdUeUXO40eb0/4f/c4uamAyj9Iri56QDh5+7QpFC5E3bKh+vr9xJ0RHdvjael/dT+/LNpP3c9rxL9IJQ/3tuCzNgIj2Elxxfu48/fq/dw68BFclTZOtM8OHED31PexAVEEhcQycmPfyMzLYOaLfXXE6rOW4MI2X2KkF2nSPEN587iHWSExVF7ch+d6WtP7E1GaBx3Fu8gxTeckF2nCNlzGvcZgzRpshJTUcUoNS+7bh7kpKuIOFBwvY8+dp2YE16k+keS6h/J/TW/kp2agXUr/ff6krbuR/bNs+TcPIs6PoKsU3tQJ8cjbdGj1PXUaUmQWuhVqLJh2KQj2ZcOkRtwE7Uyhmyv0+QE3kLWpp++w6mQLh3a8Pabk+jTvVNVZ+WZkYu6Ul/VkaiQClqOHj3K+PHjefvtt7lz5w5bt25l+/bt/O9//wPy7vIMGjSIyMhIDh8+zLVr12jZsiW9evUiPj4egF9//ZWlS5fyv//9j6tXr+Ls7MxXX31V7LtOnDjB3bt3OXXqFHv27OGPP/5g+fLlms9TU1OZO3cuV65c4cSJExgYGDBixAhNZTIlJYVu3boRHh7OX3/9hbe3N++//77OyuatW7fo1KkTL730El9//TUGBlVz6Nu6OqBwsObeWR/NsuzMbPwu3aFOqxfKvR0jEzkGMilpiSn6yGaJLGvZY+ZoRcg/NzXLcjOzCb90D6dW9Utcz7FlPa11AILP+BRbp+uqyQSd9CL03O0St+XSvhGTb2xm7JmP6b5uKia2lk8YTekkMimmHvVIPqvdNTP5nxuYtWr45Ns1kpGrytJalpuRiXnrRk+8zcoikUkxaVo85pSzNzBtVc78SSQYmJmQk5ishxxWzKP4UiojPmXVxyeRGWLRzJ34It0v48/4oGjdoNK+x9BEjkQqJesplzePmNeyx9TRirAz2uVO1MV7OLQuudyxb1WPsCLlTtgZn1LXqQrWrvZYOFjjd7YgrzmZ2QReuodrKeXq45IYSGg6pD0yEzmh1x9W2na1vkNmiKJZHWJO+2gtjznjg3Vr3dc4q9b1iTlTJP0pbxTN3ZFIDXWu4zq2OxH7L5CTptKdEQMJzsM7YGgqJ+Gq7m6zlcbAEAOn2uQGal+3cgJvY+BSemXYeOIyTKZ/hvzleRi4al9XJIZS1Nna1wqyszBwebaOX0GoDGIM6X/YwYMHMTc317wfMGAAUVFRfPjhh0yaNAkAd3d3Vq5cyfvvv8/SpUs5deoUN2/eJDo6GrlcDsAnn3zC/v372bt3L2+++SYbNmxgypQpvP766wCsWrWK48ePF2slNTIy4ocffsDU1JQmTZqwYsUK5s+fz8qVKzEwMGDUqFFa6b///nscHBy4c+cOTZs2Zffu3cTExHDlyhVsbPK6xega/3rhwgUGDx7MggULmDdvXuX9AZ+Apb0VAEkxSq3lSTFKbGrq7jKoy9APxqKMjOf++ZtlJ65Epvn5T4vVzn9ajBKLmrq7sT5aL73IOumxSkztFZr39Ya2x97Djb2Dl5S4neBT3vgdvExyWCwWrva0mzeaob8s4LeBi8nN1N1C8KQMrS2RSA3Jik3UWp4Vq8Qi/+/wJJL/uYH968NIuXSLzKBIzDs1R9G3HVTRTZLCHsWcHZOotTwrJhELO6tybcP+jeEYmMpJPHSu8jNYQZr4iuzT7NhEZOXcp3bPUHwyG0sMpIZkFilPMmOUGDlYVdr31F00DlVkPAn/PN3y5hHT/FiKlSExSsxLKXdM7K3IKPK3yYhRYlKo3HkWmOfHl1okr6mxShQuJcdXXg4NXHn9j2VI5TIyUzP4ZdrnpY5NrQijEo5JVYwSuYPuv7vcwQqVjmPYQCbFyMYCVXSi1meKF+ti2agWPnO+KbYti0audDy0AgO5jJzUDK699hkpZYwzriiJiQUSA0PUqdoxqFOTkJjpjlmdokR1dDu5kYFIpDIMG3dA/so8VD+vJzf0AQA5AbeQte5Lbsh91IkxGNRuhGG9FiCp+muF8Hiel262CQkJvP322/z1118ADB06lI0bN2JlZVXqenfv3uWDDz7gzJkz5Obm0qRJE3799Vdq1Sp/93JRIf0P69GjB19//bXmvZmZGfXq1ePKlSuaFlGAnJwcMjIySEtL49q1a6SkpGBrqz0WJD09XdOd9u7du7z11ltan3fo0IFTp05pLWvevDmmpqZaaVJSUggJCaF27dr4+fmxePFiLl68SGxsrKblMzg4mKZNm+Ll5cWLL76oqYzqEhwcTO/evVm1ahVz5swp82+iUqlQqbTvuOaoczCU6L5LW5bWwzrzyuo3NO+3Tlmb958ihZNEIim2rCS9pg2l5dBObByznOwiLW2Vrf7wjnRfO0Xz/tDkT/L+UySrefkvfVvFw5NoCmlzZxs6L5vAgXHryCklpoeFumfF3w8lxieACRc24NazBf4lTJBUYcX2FWXGWprQZd9Sa+0sGp38CtSgCoog7rfj2L7Uu2L5rFTFj8/yhGw1tCuO744l8I1V5MQpy16hqhQ9GCWScp1+iiFdcXxnLIFvPmvxFY2HcpcnZak1cyiOIzpxfeSyYi37+uI+oiMd1xWUO39P1F3ulGe/FftcIqnyH4cewzsyZPVUzftdr+WNgy9eRJb/ulCaOP9wtgxYiLGlKY0GtGX4p2+x/ZVVequUgq5dVUYsOvZT3vLi67iO7U7S3WCUOibZSnkYztmeHyJTmOE0uC3Nv5zOxREr9F4p1amUmNUJkeQk5E1epAZyw/0wsLBB2qYfmfkV0syTezDqNwnjqasBNerEGLJvnUfaVHSFfd48L499GTt2LKGhoXh65k2O9uabbzJhwgQOHDhQ4jp+fn507tyZqVOnsnz5chQKBXfv3sXYuOTx77qICul/2KMKaGG5ubksX76ckSNHFktvbGxMbm4uzs7OnD59utjnZd1BKS9J/oVoyJAhuLq68u2331KjRg1yc3Np2rSpZpypiYlJmduyt7enRo0a/Pzzz0ydOhVLy9K7d65Zs0ar2zBAW0Vj2lk1faJYbh6/SqBXQXchqZEMAEsHK5IKtUJZ2FmSFFv2D9yebwymz8zhbB63ivB7wU+Up8cR+Pd1fvEquOgbGuUVGab2CtIK3bU2sbMkLabk/KfFJGq1hj5aJz02CQD7ZnUwtVfw0uGVms8NpIbUaNcAj8l92Fp3Murc4gV6WnQiyWGxKOo4PVF8pclJSEKdnYPMXnsiF6mtolgL22NtNz6JgDdXI5HLkFpZkBUVj/OHk1CFRJW9sp49illaNGa7smNWDO5MzXVvEzRjLSnnn86slo+rxPjKsU8Vg/LiC565ltRnJL6s+CRys3MwKtK6a2SnKNZC9SRcpw+h9jsj8HppJal39F/ePBJ87DoxN4qXOyb2CtKLlDsZpZSb6TGJmBRplTO2syQjv9ypKvf/vk6YjvjM7RWkFIrPzNaSlHJcF8qSk5VDfFBe+RJ+MwCX5u60e60fBxf+UOFtF5WZf0zKi5T3RnaWqGJ0/91V0YnFWk+N7CzJzcomM0G7m7iBiRE1hnfkwfrfdG5LnZVDWmBerEpvf6xauOP2Rn9uzf/+SUMqkzo9GXVuTrHWUImpRd4Y0XLKifBH2rh9wYL0ZDL3bwJDKRITc9Qpici6jkatjK2srAuCxt27d/H09OTixYu0a9cOgG+//ZYOHTpw//59GjTQPQzko48+YuDAgaxfv16zzN398Sf9E+3+gpaWLVty//596tWrV+xlYGBAy5YtiYyMRCqVFvvczi6va1GjRo24ePGi1naLvgfw9vYmPT1dK425uTk1a9YkLi6Ou3fvsmjRInr16kWjRo1ISNCeRbBZs2Z4eXlpxq7qYmJiwsGDBzE2NqZfv34kJ5c+7mvBggUolUqtV2vFk4/tU6VmEBsUpXlF+oaijE6gQedmmjSGMkPqtmtMwLUHpW6r55tD6Dd7FFsmrSHkpv8T5+lxZKVmkBQYpXklPAgjNSqRml0KKugGMkNqtGtI5LWSx+lEXX+IaxftSr1rVw/NOqHnbvNz7w/5tf9Hmle0tz8P/viXX/t/pLMyCiC3Msfc2UarclxZ1FnZpN18iEWXFlrLLbq0IPWa7kdqPNb2VVlkRcWD1BCrAR1JOqZ7Mq6nSZ2VTfqth5h3flFruXnnFqRdu1vielZDu+L6ybsEv/MJyaeqdnKY0jxpfIohXan5ybuEPGPxqbNySPbxx6ZbM63lNl2bobxa8iM2yqPWjCHUmTsK71dXk+z9dMqbR7JTM0gOjNK8Eh+EkRaVSI2u2uWOY/uGRJcyPjDm2kNcipQ7Ll09Sl3nachMzSA+KErzivENIzk6gbqdCyaNMpQZ4tauISGllKtPTFJwc7SyqbNyUPoEYF/kmLTr6kHCVd3XuMSrvth11Z4wy757M5Te/qizc7SW1xjaHgMjKWF7y9llXiLBQE+xauTmkBsZhEFt7cfSGNZuQm5Y+cfqGjjUQp2i4wZETjbqlEQwMMTwhVbkPHx2HjkllM/zMMvuhQsXUCgUmsooQPv27VEoFPz7778618nNzeXQoUO88MIL9OvXDwcHB9q1a8f+/fsf+/tFC6mgZcmSJQwePBhXV1deeuklDAwM8PHx4ebNm6xatYrevXvToUMHhg8fzrp162jQoAHh4eEcPnyY4cOH07p1a9555x0mTZpE69at6dy5M7t27eL27dvF7phkZmYydepUFi1aRFBQEEuXLmXWrFkYGBhgbW2Nra0t33zzDc7OzgQHB/Phhx9qrf/qq6+yevVqhg8fzpo1a3B2dubGjRvUqFGDDh06aNKZmZlx6NAhBgwYwIABA/D09NQaO1uYXC7XjI195Em765bkzA+H6TNzODGBEcQERNJn5nCy0lVc+7PgAjv+05koo+I5sH4PkNdNd9Dcl/nxnS+JC43GIv/usyo1g8ySJnXQE5/vPWk1ayjKwCiUAZG0nDWU7IxMfPcXFFi9Pp9GamQCF9f9mr/OUYbvXcSL0wcTcOwadfq2ombnJvwxMq9FNCs1g/j72s9hzUpTkZGQolkuNZXTdu5I/A5fIS06EYua9rT/4CUyElL01l035rs/qfX5HNJ8HpJ6/R62r/ZDVsOe2F1HAHB+fyIyJxuC527QrGPSOO/RSQZmxhjaWmLSuA65WdmofEMAMG3xAjInW9Jv+yNzssVpzqtgICF66+96ieFxxXy3H9fP5pLu40va9XvYjO2PrIY9cfkxO70/EZmjLSHvfQ7kV0Y/nUP48m9Ju3EPaX5rXW5GJrnJumdxrUqx3+2n5mdzSb+ZH9+refHF786Lz3H+RGROtoTmx6cYkh/fivz48sfS5qqejfhCthyk8abZJHv7o7z6gBoTeiOvaUf4j38D4P7Rq8idbLg7e7NmHfMmtQEwNDNGZmuJeZPa5GZlk5bfrbHWzKG4f/AKt6d/SUZwNEb55U1OakbJk8jo2Z3vPGk2eyhJAVEkBUTSbPZQctIz8fujoNzp8sU00iISuLY2r9y58/1RBuxbhMeMwQQfvUatfq2o0aUJh0YU9MSQmsqxrOOoeW9eyx6bJrVQJaSSGh731OK7+L0nXWYOJS4wkviASLrMGkZWRiY3/yyIb8Rnb5EUmcCJ9b8AeZVW+/p5jzQxNJJi4WSNU+PamgovQK/5L+N72pukiDiMzExoOrQ9bu0bs3PiOr3FErDlEC02zSTR25/Eqw9wndALk5p2BP+Y91zRBh+NwdjJGu/ZeUOGgnYcp/bUvjRaPp6QnSexav0CrmN7cOOtjcW27Tq2B1GeV8lKKD7BVoOFrxB9wouM8Dik5ibUGN4B246NuTxmrd5ifST76lGMBr1BbmQgueF+SJt3Q2JpQ7b3aQBkXUYhsbAm83Dec6mlrfqgVsaSGxsGhlKkjTsgbdAa1f5Nmm0aOLsjMbciNzoEibkVsk7DQGJA1uUjeo+nItLS0gkODde8DwuP4t4DPxSWFjg7OVRhzqpOZc+Mq2t4ma7fr48jMjISB4fi+8fBwYHIyEid60RHR5OSksLatWtZtWoV69atw9PTk5EjR3Lq1Cm6detW7u8XFVJBS79+/Th48CArVqxg/fr1yGQyGjZsqJmgSCKRcPjwYT766COmTJlCTEwMTk5OdO3aFUfHvIv6K6+8gp+fHx988AEZGRmMGjWK6dOnc/ToUa3v6tWrF/Xr16dr166oVCrGjBnDsmXLADAwMODnn3/m7bffpmnTpjRo0IAvv/yS7t27a9Y3MjLi2LFjvPfeewwcOJDs7GwaN27M5s2bKcrc3JwjR47Qr18/Bg4cyJEjRzAzM9PPH7EMx7f8hczYiJdWTsVUYUaQ10O+mrAaVWrBpE/WLrao1QWzBXee0AepXMbULe9pbevIht84smHvU8s7wI2vDyI1NqLrqsnIFaZEeflxYNw6sgrl39zFTusuXuQ1X47N3ES7+S/Rdt5okoKi+HvGJqK9io8BKok6Nxebhq68MKozcksz0qITCbtwh2MzNml9d2VKPHgOQ2sLnN5+BamDDRkPgvCfvIKssBgAZA7WGBV5fmWDI19o/m/arD42w7uTGRLFnc55Y4klchnO88Zh5OpEbloGSaeuEvTu5+SU8JiYp0158BxSK0sc3xmD1D4v5sDXlmtiljrYIHMpiNlmbH8kMikuq6bjsmq6Znn83hOEztvwtLNfJuWhcxhaW+Lwdl58qgdBBE5ZXmif2mg9k1QT38rpuKwsiC9h7wlC52942tkvJvrPC8isLXCbOwq5ozUp90LwGbuGjNC8bn1yB2uMi0yM0/ZkwbN7LVvUxWlUF9KDo7nQZhYALpP7YiCX4fGDdnkT8PFvBHyiu6ukvt386iCGxkZ0WD0ZI4UpsTf8ODp2HdmFzn2zGnZavSmir/pyesYmWr7/Ei/OH01yUBSnp28itlB3Wbvm7gzY+5Hmfbtl4wHw/fUfzumYNEdfzm85iMzYiEGrJmNiaUaolx8/jV9LZqH4FDVsteKzcLTmrSOrNe87TRtMp2mDCbxwh+1j8uaBMLNXMPLz6Zg7WKFKTiPqXgg7J67D/9wtvcUS8edFjKwtqD93JHJHK1LuhXBl7DrSNcekFSaFjsn04BiujF1P4xUTqP1aX1RRCdz+6EciD13W2q6ZuxM27Rty6aXV6CK3V9Bi00zkjlZkJ6eRfCeYy2PWEvsUJuPKuX+FLBNzZB2HIjFToI4NQ7VvA+qkvJsaEnOF9jNJDQ2RdX8Zibk1ZGeSGxdOxt7PyQ0olFdDKbLOI5FY2UNmBjkBN8k89B2o0nmW3brny5TZH2jer9+Ydx4NG9Cb/y16r6TVhMega3jZ0qVLNb+hC1u2bFmxtEVduXIFKBgyV5harda5HNDM7TJs2DDNPC0tWrTg33//ZcuWLY9VIZWoq3p0vyA84952e6Wqs6BXDXP03J2pinU0qNrxYk+DgaR6F+MlXAurlZi0ssfEP8+CpEZVnQW9CpJW73MQoE1G1T2/+2noMa96n4Oy1xZVdRb0Tmb3+GMXnwZLs8rNV0z83XK3kMbGxhIbW/q4Yzc3N3bv3s3cuXNJTEzU+szKyorPP/+c1157rdh6mZmZmJmZsXTpUhYtKji+PvjgA86dO8f58+fLHZNoIRUEQRAEQRAEQdCDyp5l93G659rZ2WnmeClNhw4dUCqVXL58mbZt2wJw6dIllEolHTt21LmOkZERbdq04f597TkLHjx4QO3atcuVv0fEpEaCIAiCIAiCIAj/UY0aNaJ///688cYbXLx4kYsXL/LGG28wePBgrRl2GzZsyB9//KF5P3/+fH755Re+/fZbHj58yKZNmzhw4AAzZsx4rO8XFVJBEARBEARBEAQ9UFfyP33ZtWsXHh4e9O3bl759+9KsWTN++uknrTT3799HqSyYDXrEiBFs2bKF9evX4+HhwXfffce+ffvo3LnzY3236LIrCIIgCIIgCIKgB5XdZVdfbGxs2LlzZ6lpdE09NGXKFKZMmVKh7xYtpIIgCIIgCIIgCEKVEC2kgiAIgiAIgiAIeiAeaFI2USEVBEEQBEEQBEHQA32O+6wuRJddQRAEQRAEQRAEoUqIFlJBEARBEARBEAQ9EF12yyYqpIIgCIIgCIIgCHogKqRlE112BUEQBEEQBEEQhCohWkgFQRAEQRAEQRD0QLSPloNaEIRnRkZGhnrp0qXqjIyMqs6KXlT3+NTq6h+jiO/5V91jFPE9/6p7jNU9PrX6vxGjUHkkarXo2CwIz4qkpCQUCgVKpRJLS8uqzk6lq+7xQfWPUcT3/KvuMYr4nn/VPcbqHh/8N2IUKo8YQyoIgiAIgiAIgiBUCVEhFQRBEARBEARBEKqEqJAKgiAIgiAIgiAIVUJUSAXhGSKXy1m6dClyubyqs6IX1T0+qP4xivief9U9RhHf86+6x1jd44P/RoxC5RGTGgmCIAiCIAiCIAhVQrSQCoIgCIIgCIIgCFVCVEgFQRAEQRAEQRCEKiEqpIIgCIIgCIIgCEKVEBVSQRAEQRAEQRAEoUqICqkgCHqjVqsJCgoiPT29qrPyVCUmJlZ1FoTHlJmZyf3798nOzq7qrOhdUlIS+/fv5+7du1WdlUqTnZ3N8ePH2bp1K8nJyQCEh4eTkpJSxTkTBEEQyiIqpILwDKiuP4bVajX169cnNDS0qrOiN+vWreOXX37RvH/55ZextbXFxcUFb2/vKsyZUB5paWlMnToVU1NTmjRpQnBwMABvv/02a9eureLcVY6XX36ZTZs2AZCenk7r1q15+eWXadasGfv27avi3FVcUFAQHh4eDBs2jJkzZxITEwPA+vXrmTdvXhXnThDyuLu7ExcXV2x5YmIi7u7uVZAjQXh2iAqpIFSh6v5j2MDAgPr16+u8CFcXW7duxdXVFYC///6bv//+myNHjjBgwADmz59fxbmrHKmpqSxevJiOHTtSr1493N3dtV7PswULFuDt7c3p06cxNjbWLO/du7fWjYbn2T///EOXLl0A+OOPP1Cr1SQmJvLll1+yatWqKs5dxb3zzju0bt2ahIQETExMNMtHjBjBiRMnqjBnglAgMDCQnJycYstVKhVhYWFVkKOn67/WU0p4PNKqzoAg/JcV/jHcv39/zfLevXuzdOlSPvzwwyrMXeVYv3498+fP5+uvv6Zp06ZVnZ1KFxERoamQHjx4kJdffpm+ffvi5uZGu3btqjh3leP111/nzJkzTJgwAWdnZyQSSVVnqdLs37+fX375hfbt22vF1bhxY/z8/KowZ5VHqVRiY2MDgKenJ6NGjcLU1JRBgwZVi5sm586d4/z58xgZGWktr127drX7oX/27Fm2bt2Kn58fe/fuxcXFhZ9++ok6derQuXPnqs5ehVXH+P766y/N/48ePYpCodC8z8nJ4cSJE7i5uVVBzirfzJkz2bx5c7HlqampDBo0iNOnTz/9TAnPBVEhFYQq9F/4MTx+/HjS0tJo3rw5RkZGWi0YAPHx8VWUs8phbW1NSEgIrq6ueHp6alqc1Gq1zrvhz6MjR45w6NAhOnXqVNVZqXQxMTE4ODgUW56amlptKt6urq5cuHABGxsbPD09+fnnnwFISEjQahV+XuXm5uo810JDQ7GwsKiCHOnHvn37mDBhAuPGjePGjRuoVCoAkpOTWb16NYcPH67iHFZMdY1v+PDhAEgkEiZNmqT1mUwmw83NjU8//bQKclb5jh07xqJFi7R6XqSmpmrdcBcEXUSFVBCq0H/hx/CGDRuqOgt6NXLkSMaOHavpmjxgwAAAvLy8qFevXhXnrnJYW1trWtiqmzZt2nDo0CFmz54NoDnvvv32Wzp06FCVWas07777LuPGjcPc3JzatWvTvXt3IK8rr4eHR9VmrhL06dOHDRs28M033wB5+zAlJYWlS5cycODAKs5d5Vm1ahVbtmxh4sSJmpsKAB07dmTFihVVmLPKUV3jy83NBaBOnTpcuXIFOzu7Ks6R/hw7dozOnTtja2vLnDlzSE5Opl+/fkilUo4cOVLV2ROeYaJCKghV6L/wY7joHeHq5vPPP8fNzY2QkBDWr1+Pubk5kNeVd8aMGVWcu8qxcuVKlixZwo8//oipqWlVZ6dSrVmzhv79+3Pnzh2ys7P54osvuH37NhcuXODMmTNVnb1KMWPGDNq2bUtISAh9+vTBwCBv+gh3d/dqMYb0888/p0ePHjRu3JiMjAzGjh2Lr68vdnZ27Nmzp6qzV2nu379P165diy23tLSsFjN7V/f4AgICqjoLelenTh2OHj1K9+7dMTAw4Oeff0Yul3Po0CHMzMyqOnvCM0xUSAWhCv0XfgwD+Pn5sW3bNvz8/Pjiiy9wcHDA09MTV1dXmjRpUtXZqxCZTKZzJs9333336WdGTz799FP8/PxwdHTEzc0NmUym9fn169erKGcV17FjR86fP88nn3xC3bp1OXbsGC1btuTChQvVovXwkdatW9O6dWutZYMGDaqi3FSuGjVq4OXlxZ49e7h+/Tq5ublMnTqVcePGFRsi8Dxzdnbm4cOHxcYbnjt37rmfXAyqf3xltfIuWbLkKeVEv5o2bcrBgwfp3bs37dq14+DBg9XqPBT0Q6JWq9VVnQlB+C+7efMmn3zyCdeuXSM3N5eWLVvywQcfVJsfw2fOnGHAgAF06tSJf/75h7t37+Lu7s769eu5fPkye/fureosVsiOHTtK/XzixIlPKSf6s3z58lI/X7p06VPKifAkcnJy2L59OydOnCA6OlrThfCRkydPVlHOhMexfv16fvzxR3744Qf69OnD4cOHCQoKYs6cOSxZsoRZs2ZVdRYrpLrH9+KLL2q9z8rKIiAgAKlUSt26dZ/bG3svvviiziFGQUFBODg4aFVGn9cYBf0TFVJBEPSqQ4cOvPTSS8ydOxcLCwu8vb1xd3fnypUrDB8+/LmfBdPa2lrrfVZWFmlpaRgZGWFqavrcT9r0XxEdHa2zstasWbMqylHlmTVrFtu3b2fQoEE6Z0n+/PPPqyhnlcPQ0JCuXbuyb98+rbHOUVFR1KhRo9pMLgbw0Ucf8fnnn5ORkQGAXC5n3rx5rFy5sopzVjmqe3xFJSUlMXnyZEaMGMGECROqOjtPpKwbloWJm5dCSUSFVBCqUFJSks7lEokEuVxe7DEGzyNzc3Nu3rxJnTp1tCqkgYH/b+/Oo6qq+v+Bv89lVgFBgURlEiccEnOecp4TxSchJxR8/Gom5oyZqaRFzhIamqGSIyWWljOoJWoKAg4MKog44YAiCch4f3/48z5eUTO51809vF9rtZZ3n/PH+yxS7ufsvT87DQ0aNFB98ZCTS5cuYfz48Zg+fTp69eolOo7GxMTEIDExEZIkwcXFpdQbf10UExMDLy8vJCYm4vlfh5IkyaKYqV69OkJDQ2XV4OdZCoUCbdq0QUZGBnbt2qU6Xur27duoUaNGqZcMui43NxcJCQkoKSmBi4uLat+6XMj9+Z53/vx59O/fH2lpaaKjEAnDPaREAlWtWvWV3XRr1aqFUaNGYe7cuapGJLqmatWquHXrFhwdHdXGY2NjUbNmTUGptKtu3boICAjA8OHDkZSUJDpOmd25cweenp44cuQIqlatCqVSiYcPH6JLly7Ytm0brKysREd8Y6NHj0a9evXwww8/wMbGRjbdrZ9laGgom47PLyJJEnbs2IGAgAC0a9cOP/74I9zc3FTX5KZSpUql9gPLidyf73lZWVl4+PCh6Bgacfr0aZSUlJQ6g/uvv/6Cnp5ehfq50r/DgpRIoA0bNmD27NkYNWoUWrVqBaVSidOnT2Pjxo34/PPPcffuXSxZsgRGRkb47LPPRMd9I0OHDsXMmTPx008/QZIklJSUICoqCtOmTZPF/sqX0dPTw82bN0XH0IiJEyciOzsbFy5cQMOGDQEACQkJ8PLygq+vr053Mr1y5QrCw8NlXbBNnToVK1euRFBQkCwLNKVSCT09PaxcuRKNGjWCh4cHPv/8c4wZM0Z0tDJzd3d/7XvDw8O1mEQ75P58zwoMDFT7rFQqcevWLfz444+yOadzwoQJmDFjRqmC9MaNG/jmm2/w119/CUpG5R0LUiKBNm7ciKVLl2LIkCGqsQEDBqBJkyZYs2YNIiIiYGdnh4ULF+psQbpw4UKMGjUKNWvWhFKphIuLC4qLizF06FB8/vnnouOV2a5du9Q+P/2SERQUhPbt2wtKpVn79u3DoUOHVMUoALi4uGDVqlXo2bOnwGRl161bN8THx8u6ID127BgOHz6MvXv3olGjRqW6JOv6F/1njR07FvXq1cN//vMfWXQqNzc3V/1ZqVRi586dMDc3V800xcTEICsr618VduWJ3J/vWc/v1VYoFLCysoKXlxdmzZolKJVmJSQkoHnz5qXGXV1dkZCQICAR6QoWpEQCnThxAsHBwaXGXV1dceLECQBAhw4dkJ6e/rajaYyBgQE2b94Mf39/xMbGoqSkBK6urqhbt67oaBoxcOBAtc+SJMHKygpdu3bF0qVLxYTSsJKSklJFDPDkZ6vr+/PWrVsHLy8vnD9/Ho0bNy71nAMGDBCUTHOqVq2KQYMGiY6hNfb29tDT01N97ty5M06ePIkPPvhAYCrNWL9+verPM2fOxJAhQxAcHKx63uLiYnz88ccwMzMTFbFM5P58z6oI55AaGRnh9u3bpY7puXXrFvT1WXLQy7GpEZFA9erVg7u7OwICAtTG/fz8sHPnTiQnJyM6Ohpubm46342WdJebmxuysrKwdetW2NraAniyBGvYsGGwsLDAzp07BSd8c7t27cKIESPw999/l7oml6ZGFdXjx49x+/Zt2Nvbi46iEVZWVjh27Bjq16+vNp6cnIx27dohMzNTUDLNkPvzPevatWuQJAm1atUSHUWjPD09kZGRgV9//VU1+52VlYWBAwfC2toaYWFhghNSecXXFUQCLVmyBB9++CH27t2Lli1bQpIknD59GomJidixYweAJ00CPDw8BCf9d6ZMmfLa9y5btkyLSbTP398f06ZNQ6VKldTG8/LysHjxYlkcdh4UFAQ3Nzc4ODigdu3akCQJ6enpaNKkCTZt2iQ6Xpn4+vpixIgRmDNnDmxsbETH0aq7d+8iOTkZkiShXr16Ot2M6nUYGxvLphgFgKKiIiQmJpYq2BITE3V+pQJQMZ5v/vz5CAwMxKNHjwA86UI/ceJEzJ0794WrUHTN0qVL0alTJ9jb26u6sMfFxcHGxgY//vij4HRUnnGGlEiwq1ev4rvvvsPFixehVCrRoEED/N///R+ysrLQrFkz0fHeSJcuXdQ+x8TEoLi4WPVF4+LFi9DT08N7772HyMhIERE1Rk9PD7du3YK1tbXaeGZmJqytrWU1w3bw4EEkJSWp9gJ3795ddKQyMzU1RVxcHOrUqSM6itbk5ORg4sSJCA0NVX2x19PTw8iRI/Htt9+WepmiCywtLXHx4kVUr14dFhYWr2zWJJezgKdMmYINGzbgs88+Q5s2bQAAJ0+eREBAAEaOHKnzL/fk/nzjxo3Dzp074e/vj7Zt2wJ4sm1n3rx5cHNze+H2HV2Uk5ODzZs3Iz4+HiYmJmjatCk++ugjWRTcpD0sSInKkaysLGzevBkhISGIi4uTRTGzbNkyHDlyBBs3boSFhQUA4MGDBxg9ejQ6duyIqVOnCk5YNgqFArdv3y412xQZGQkPDw/cvXtXUDJ6HV5eXujYsaMsOrK+zP/93//h0KFDao22jh07Bl9fX/To0QPfffed4IT/3saNG+Hp6QkjIyNs2LDhlQWpl5fXW0ymPSUlJViyZAlWrlyJW7duAQBq1KiBSZMmYerUqWr7aHWR3J/P3Nwc27ZtQ58+fdTG9+7dC09PT9kc/UL0JliQEpUDkZGRCAkJQXh4OOzt7TF48GAMHjxYteRFl9WsWRMHDhxAo0aN1MbPnz+Pnj176uzRKE9nZR4+fAgzMzO1L8TFxcV49OgRxo0bh1WrVglM+eYCAwMxduxYGBsblzqu4Hm+vr5vKZXmLVy4ECtWrEC/fv3QpEmTUm/xdfnZnqpevTp+/vlndO7cWW388OHDGDJkCF+a6KDs7GwAkEWznxeR4/PZ2NjgyJEjat3KgSdLkjt16iSrv4cJCQlIT09HQUGB2rgcmsSRdrAgJRLk+vXr2LBhA0JCQpCTk6PqLhgfHw8XFxfR8TTG1NQUv/76K7p27ao2HhkZCTc3txc2k9EFGzduhFKphLe3N1asWKF2fIGhoSEcHBxUy7J0kaOjI6Kjo1GtWjU4Ojq+9D5JkpCamvoWk2mWnJ/tqUqVKiEmJqbUF+ELFy6gVatWyMnJEZRMM86cOQMDAwM0adIEAPDrr79i/fr1cHFxwbx582BoaCg4IdGTfgNJSUlYv349jIyMAAD5+fnw8fFB3bp1MXfuXMEJyy41NRWDBg3CuXPnIEkSnpYYT1/YymHVF2kHC1IiAfr27Ytjx46hf//+GDZsGHr37g09PT0YGBjIriAdOXIkjh49iqVLl6rtC5o+fTo6deqEjRs3Ck5YNkePHkX79u3Z0p7KrW7duqFatWoIDQ2FsbExgCdNt7y8vHD//n0cOnRIcMKyadmyJfz8/DB48GCkpqbCxcUF7u7uOH36NPr164cVK1aIjqgRjo6Or1yarOsvT+T+fIMGDUJERASMjIzw7rvvAgDi4+NRUFCAbt26qd2rq2cDf/DBB9DT08P3338PJycnnDp1CpmZmZg6dSqWLFmCjh07io5I5RS/QREJcODAAfj6+mL8+PGyOY/zZYKDgzFt2jQMHz4chYWFAAB9fX34+Phg8eLFgtOVXdeuXWXf1KgidBIGUOptvlysXLkSvXv3Rq1atfDuu+9CkiTExcXB2NgY+/fvFx2vzC5evKhqAPfTTz/h/fffx5YtWxAVFQVPT0/ZFKSffvqp2ufCwkLExsZi3759mD59uphQGiT356tatSoGDx6sNla7dm1BabTjxIkTiIyMhJWVFRQKBRQKBTp06ICvv/4avr6+iI2NFR2RyinOkBIJcOLECYSEhCAsLAwNGjTAiBEj4OHhAVtbW9nNkD6Vk5ODlJQUKJVKODs7o3LlyqIjaYRCoUBGRkapgvTmzZuoU6cO8vLyBCXTHLl3Eg4NDcXixYtx6dIlAE/OB54+fTpGjBghOJnm5OXlYdOmTWpdkocNGwYTExPR0crMzMwMMTExqFu3Lnr06IH+/ftj0qRJSE9PR/369WXxd/BVVq1ahejoaKxfv150FK2Q+/PJiYWFBWJiYuDk5IQ6depg3bp16NKlC1JSUtCkSRPk5uaKjkjlFGdIiQRo27Yt2rZti5UrV2Lbtm0ICQnBlClTUFJSgoMHD6J27dowNTUVHVOjKleujKZNm4qOoTFPG/1IkoR169ahSpUqqmvFxcX4448/0KBBA1HxNEqpVL5w1jA+Ph6WlpYCEmnOsmXLMGfOHHzyySdo3749lEoloqKiMG7cONy7dw+TJ08WHVEjTExM8N///ld0DK1o0aIFFixYgO7du+Po0aOqrsFXrlyR/dmyANCnTx/MmjVLtgWbXJ6va9euCA8PR9WqVdXGs7OzMXDgQJ0/Ag0AGjdujLNnz8LJyQmtW7fGokWLYGhoiLVr18LJyUl0PCrHOENKVE4kJyfjhx9+wI8//oisrCz06NEDu3btEh2rzHJychAQEICIiAjcuXOn1AHnurov6GkznKtXr6JWrVpqRxI8bWrk7++P1q1bi4pYZnLvJAw8+TnOnz8fI0eOVBvfuHEj5s2bhytXrghKVja7du1Cnz59YGBg8I//juh658uzZ89i2LBhSE9Px5QpU1TNYSZOnIjMzExs2bJFcELtWrRoEVavXo20tDTRUbRCLs/3stU0d+7cQc2aNVVbWnTZ/v37kZOTA3d3d6SmpqJ///5ISkpCtWrVsH379lLNDYmeYkFKVM4UFxdj9+7dCAkJkUVB+tFHH+Ho0aMYMWIEatSoUWqmbdKkSYKSaUaXLl0QHh6uOmNVTuTeSRgAjI2Ncf78eTg7O6uNX7p0CU2aNMHjx48FJSubZ7/8KhSKl94nSZLOL7l+mcePH6uaxcmBq6ur2r+fSqUSGRkZuHv3LlavXo2xY8cKTFd2cn2+s2fPAgCaNWuGyMhItVUlxcXF2LdvH9asWaPzBffL3L9/X/Vyk+hlWJASkVZVrVoVv//+O9q3by86Cr2ho0ePol27drL5Yv+sxo0bY+jQofjss8/UxhcsWIDt27fj3LlzgpIRqZs3b57al3qFQgErKyt07txZFtsD5Pp8CoVC9Vwv+sptYmKCb7/9Ft7e3m87mlZdu3YNkiShVq1aoqOQDmBBSkRa5ejoiD179pQ6A1FOrl+/jl27dr3wIPBly5YJSqUdeXl5pZaW6fLh9Tt27ICHhwe6d++O9u3bQ5IkHDt2DBEREQgLC8OgQYNERyyz0NBQeHh4qM4+fKqgoADbtm0rtVyZiDTn6tWrUCqVqmNQrKysVNcMDQ1hbW2ttuVDlxUVFWH+/PkIDAzEo0ePAABVqlTBxIkTMXfuXFm+1CTNYEFKRFq1adMm/Prrr9i4cWOpY0PkICIiAgMGDICjoyOSk5PRuHFjpKWlQalUonnz5rJoVJGbm4sZM2YgLCwMmZmZpa7r+pLPmJgYLF++HImJiaoOtFOnToWrq6voaBoh9y7JFYXcf45yf76KYNy4cdi5cyf8/f1V2zlOnDiBefPmwc3NDcHBwYITUnnFLrtEpFVLly5FSkoKbGxs4ODgUOoN6ZkzZwQl04xZs2Zh6tSp8Pf3h6mpKXbs2AFra2sMGzYMvXv3Fh1PI6ZPn47Dhw9j9erVGDlyJFatWoUbN25gzZo1CAgIEB2vzN577z1s2rRJdAyteVmX5OvXr6vtC6by7WXzB/n5+TA0NHzLaTRP7s8XGhr6yutyWKmwdetWbNu2DX369FGNNW3aFHZ2dvD09GRBSi/FgpSItGrgwIGiI2hVYmIitm7dCgDQ19dHXl4eqlSpAn9/f7i5uWH8+PGCE5bd7t27ERoais6dO8Pb2xsdO3aEs7Mz7O3tsXnzZgwbNkx0xDe2Z88e6OnpoVevXmrj+/fvR0lJidoXK13ztEmMJEno1q0b9PX/9yu/uLgYV65ckcVLkyNHjqBz586iY2iN3I+YkvvzPfV8A7/CwkLk5ubC0NAQlSpVkkVBamxsDAcHh1LjDg4OsnipQNrDgpSItOrpEQxyVblyZeTn5wMAbG1tkZKSgkaNGgEA7t27JzKaxty/f191zI2ZmRnu378PAOjQoYPOF9x+fn4vnOVVKpXw8/PT6YL06cuguLg49OrVS+2L/tMuyYMHDxaUTnN69+6NmjVrYvTo0fDy8kLt2rVFR9Ko5cuXA3jy/2RwcPALj5jS5ZknuT/fUw8ePCg1dunSJYwfPx7Tp08XkEjzJkyYgC+//BLr169X7VnPz8/HwoUL8cknnwhOR+UZC1Ii0rqsrCz8/PPPSElJwfTp02FpaYkzZ87AxsYGNWvWFB2vTNq0aYOoqCi4uLigX79+mDp1Ks6dO4fw8HC0adNGdDyNcHJyQlpaGuzt7eHi4oKwsDC0atUKu3fvLnXIu665dOkSXFxcSo03aNAAly9fFpBIc56+DHJwcICHhweMjY0FJ9KOmzdvYtOmTdiwYQPmzZuHbt26wcfHBwMHDpTFrMzTs3DlesSU3J/vVerWrYuAgAAMHz4cSUlJouO8EXd3d7XPhw4dQq1atfDuu+8CAOLj41FQUIBu3bqJiEc6gk2NiEirzp49i+7du8Pc3BxpaWlITk6Gk5MT5syZg6tXr/7jvpryLjU1FY8ePULTpk2Rm5uLadOm4dixY3B2dsby5cthb28vOmKZLV++HHp6evD19cXhw4fRr18/FBcXo6ioCMuWLdPps2TfeecdbNmypdSB7YcOHcLQoUNx584dQcnoTcTFxSEkJARbt25FSUkJhg0bBh8fH9WXY6LyJjY2Fu+//z6ys7NFR3kjo0ePfu17169fr8UkpMtYkBKRVnXv3h3NmzfHokWLYGpqivj4eDg5OeH48eMYOnSobA8Dl7OrV68iJiYGderU0fkv+mPHjsXJkyexc+dO1KlTBwBw+fJlDB48GC1btsS6desEJyy74uJiLF++HGFhYS88mujpEmy5uHnzJtauXYuAgADo6+vj8ePHaNu2LYKDg1XL6XXFlClT8OWXX6Jy5cqYMmXKK+/VxSOm5P58z9q1a5faZ6VSiVu3biEoKAi1a9fG3r17BSUjEo9LdolIq06fPo01a9aUGq9ZsyYyMjIEJNIsJycnnD59GtWqVVMbz8rKQvPmzZGamioomfbY29vLYuYXABYvXozevXujQYMGqgPcr1+/jo4dO2LJkiWC02nG/PnzsW7dOkyZMgVz5szB7NmzkZaWhl9++QVffPGF6HgaUVhYiF9//RUhISE4ePAgWrRogaCgIHz00Ue4f/8+Zs6ciQ8//BAJCQmio/4rsbGxqnN/z5w588Juybrs2eeLjY196X1yeO7nG/xJkgQrKyt07doVS5cuFROKqJzgDCkRaZWNjQ327dsHV1dXtRnSAwcOwMfHB9euXRMdsUwUCgUyMjJKnZ13+/Zt2NnZqRoe6bqIiAjVWZ2SJKFBgwb49NNP0b17d9HRykypVOLgwYOIj4+HiYkJmjZtik6dOomOpTF16tRBYGAg+vXrB1NTU8TFxanGTp48iS1btoiOWCYTJ05UdboePnw4xowZg8aNG6vdk56eDgcHB5SUlIiISKRy9+5dSJKE6tWri46iEU+7eb8OXT/mjbSHM6REpFVubm7w9/dHWFgYgCdvhdPT0+Hn56fTHT6fXX61f/9+tfMci4uLERER8cL297ooKCgIkydPxn/+8x/VftGTJ0+ib9++WLZsmc53T5QkCT179kTPnj1FR9GKjIwMNGnSBABQpUoVPHz4EADQv39/zJkzR2Q0jUhISMC3336LwYMHv7SJka2tLQ4fPvyWk2mWt7c3Vq5cCVNTU7XxnJwcTJw4ESEhIYKS0T/JysrC7NmzsX37dlW3XQsLC3h6emLBggU63Rzu2Znfx48fY/Xq1XBxcUHbtm0BPPldceHCBXz88ceCEpIu4AwpEWlVdnY2+vbtiwsXLuDvv/+Gra0tMjIy0KZNG+zduxeVK1cWHfGNKBQKAE+Kmef/GTUwMICDgwOWLl2K/v37i4inUTVr1sSsWbNKFZ6rVq3CwoULcfPmTUHJNCMiIgIRERG4c+dOqRk0OXzJr1+/PkJDQ9G6dWt07NgR/fr1g5+fH7Zv346JEyeycZOO0NPTw61bt0qtxrh37x7eeecdFBUVCUqmGYMGDXrhTJskSTA2NoazszOGDh2K+vXrC0j35u7fv4+2bdvixo0bGDZsGBo2bAilUonExERs2bIFtWvXxvHjx2XRXXjMmDGoUaMGvvzyS7XxuXPn4tq1a7L495S0gwUpEb0Vhw8fRkxMDEpKStC8eXNZLPUEAEdHR5w+fVo2y69exNTUFLGxsXB2dlYbv3TpElxdXfHo0SNBycpu/vz58Pf3R4sWLVCjRo1SX4h37twpKJnm+Pn5wczMDJ999hl+/vlnfPTRR3BwcEB6ejomT578wnNYy7vnG8S8yoABA7SYRPuys7OhVCphYWGBS5cuwcrKSnWtuLgYu3fvhp+fn86/GBo1ahR++eUXVK1aFe+99x6USiViY2ORlZWFnj17Ij4+HmlpaYiIiED79u1Fx31tn376KSIiInDo0CHY2NioXcvIyEDPnj3RrVs31Xmsuszc3BzR0dGoW7eu2vilS5fQokUL1eoMouexICUircjLy0NERIRqhnDWrFlq+yn19fXh7+8vy7MRs7KydHoJ1vOGDRuGZs2alTq8fcmSJYiJiVHt39NFNWrUwKJFizBixAjRUd6av/76C1FRUXB2dtbZYu3pCoV/IkkSiouLtZxGuxQKxSv36EmShPnz52P27NlvMZXm+fn5ITs7G0FBQaqfb0lJCSZNmgRTU1MsXLgQ48aNw4ULF3Ds2DHBaV+fg4MD1qxZg169er3w+r59+zBu3DhZdJx/55138PXXX5c6Cmb9+vXw8/PD7du3BSWj8o4FKRFpxZo1a/Dbb79h9+7dAJ7MsjVq1AgmJiYAgKSkJMyYMQOTJ08WGbPMvvnmGzg4OMDDwwMA8OGHH2LHjh2oUaMG9uzZo/PHogDAggULsGTJErRv315tX1BUVBSmTp0KMzMz1b2+vr6iYr6RatWq4dSpU6ojX4jKm6NHj0KpVKJr167YsWMHLC0tVdcMDQ1hb28PW1tbgQk1w8rKClFRUahXr57a+MWLF9GuXTvcu3cP586dQ8eOHZGVlSUm5BswMjJCSkqKqov3865fvw5nZ2c8fvz4LSfTvICAAMybNw9jxoxBmzZtADz5XRESEoIvvvgCfn5+ghNSecWClIi0olOnTpg8eTIGDRoEAGoddgFg06ZNWLVqFU6cOCEyZpk5OTlh06ZNaNeuHQ4ePIghQ4Zg+/btqjMfDxw4IDpimTk6Or7WfZIk6dwxNzNnzkSVKlVk0dznZb7++mvY2NjA29tbbTwkJAR3797FzJkzBSWjf+Pq1auoXbv2a88O6xoLCwts3Lix1Kz9rl274OXlhQcPHuDSpUto1aqVqjGQLqhZsya2b9+ODh06vPD6n3/+CU9PT9y4ceMtJ9OOsLAwrFy5EomJiQCAhg0bYtKkSRgyZIjgZFSescsuEWnFxYsX1d50Gxsbq32RatWqFSZMmCAimkbdunULtWvXBgD89ttvGDJkCHr27AkHBwe0bt1acDrNuHLliugIWvP48WOsXbsWhw4dQtOmTWFgYKB2fdmyZYKSac6aNWteeLRLo0aN4OnpqZMFaWBgIMaOHQtjY2MEBga+8l5dm7V/madn/+bm5iI9PR0FBQVq15s2bSoilsaMGDECPj4++Oyzz9CyZUtIkoRTp07hq6++wsiRIwE8mS1u1KiR4KT/Tu/evTF79mwcPHiwVBfo/Px8zJkzB7179xaUTvOGDBnC4pP+Nc6QEpFWmJiYIC4u7qUdEZOSktCsWTOdX6Zka2uLn3/+Ge3atUP9+vWxYMECfPjhh0hOTkbLli2RnZ0tOiK9QpcuXV56TZIkREZGvsU02mFsbIzExMRSM92pqalwcXHRyb+Djo6OiI6ORrVq1V45g6+Ls/Yvc/fuXYwePRp79+594XVd3ytbXFyMgIAABAUFqfYa2tjYYOLEiZg5cyb09PSQnp4OhULx0uWv5dH169fRokULGBkZYcKECWjQoAGAJ8cVrV69Gvn5+YiOjla92NR1WVlZ+Pnnn5Gamopp06bB0tISZ86cgY2NDWrWrCk6HpVTnCElIq2oVasWzp8//9KC9OzZszr1peJl3N3dMXToUNStWxeZmZno06cPACAuLq5UV1pd9fxSz+fpcit/XT+b8nXUrl0bUVFRpQq3qKgond17+OysvZxn8J/16aef4sGDBzh58iS6dOmCnTt34vbt21iwYAGWLl0qOl6Z6enpYfbs2Zg9e7bqRd6z+9MBwM7OTkS0MqlVqxZOnDiBjz/+GLNmzVIdEyZJEnr06IGgoCDZFKNnz55F9+7dYW5ujrS0NIwZMwaWlpbYuXMnrl69itDQUNERqZxiQUpEWtG3b1988cUX6NevX6lOunl5eZg/fz769esnKJ3mLF++HA4ODrh27RoWLVqEKlWqAHiylFcuB4E/v1+rsLAQ58+fR1ZWFrp27SooFb2uMWPG4NNPP0VhYaHq5xUREYEZM2Zg6tSpgtPR64qMjMSvv/6Kli1bQqFQwN7eHj169ICZmRm+/vprWfx7+tTzhaiuc3R0xN69e1X7YAHA2dlZrUGVHEyZMgWjRo3CokWLYGpqqhrv06cPhg4dKjAZlXdcsktEWnH79m00a9YMhoaG+OSTT1CvXj1IkoSkpCQEBQWhqKgIsbGxpc5lI91QUlKCjz/+GE5OTpgxY4boOP+au7v7a90XHh6u5STap1Qq4efnh8DAQNW+Q2NjY8ycORNffPGF4HRlp1Qq8fPPP+Pw4cO4c+cOSkpK1K7L4WcIPCnSzp49CwcHBzg4OGDz5s1o3749rly5gkaNGiE3N1d0xDK5ffs2pk2bhoiICNy5cwfPfz3V9SXJFYG5uTnOnDmDOnXqqDUyvHr1KurXr6+T2wPo7eAMKRFphY2NDY4fP47x48fDz8+v1DKl1atXsxjVYQqFApMnT0bnzp11siA1NzcXHeGtkSQJ33zzDebMmYPExESYmJigbt26MDIyEh1NIyZNmoS1a9eiS5cusLGxeeWZnbqsfv36SE5OhoODA5o1a4Y1a9bAwcEBwcHBqFGjhuh4ZTZq1Cikp6djzpw5qFGjhmx/jnJmbGz8wr4JycnJsLKyEpCIdAVnSIlI6+7fv4/Lly8DkOcypYpqz5498PLywt27d0VHoQrM0tISmzZtQt++fUVH0arNmzejsLAQo0aNQmxsLHr16oV79+7B0NAQGzduVJ2FrKtMTU3x559/olmzZqKj0BsaO3Ys7t69i7CwMFhaWuLs2bPQ09PDwIED0alTJ6xYsUJ0RCqnWJASEdErTZkyRe2zUqnErVu38Pvvv8PLywtBQUGCktHLuLu7Y8OGDTAzM/vH5cm6vqT16f68p91LK4rc3FwkJSXBzs4O1atXFx2nzFxcXLB582a4urqKjkJvKDs7G3379sWFCxfw999/w9bWFhkZGWjbti327NmDypUri45I5RSX7BIR0SvFxsaqfVYoFLCyssLSpUv/sQMviWFubq5a8mhmZibr5Y/z5s3D/PnzERISAhMTE9FxNOr5l0Gvoutn5q5YsQJ+fn6qpcike8zMzHDs2DFERkbizJkzKCkpQfPmzdG9e3fR0aic4wwpERGRzOzatQt9+vSBgYGB6Chal5ubC3d3d0RFRcHBwaHUM585c0ZQsrJ71Tm5z5LDmbkWFhbIzc1FUVERKlWqVOrneP/+fUHJ6E08fvwYRkZGsn4ZRprDGVIion/JwsLitX/JyuFL1JUrV1BUVIS6deuqjV+6dAkGBgaczSiHBg0ahIyMDFhZWUFPTw+3bt2CtbW16FhaMWrUKMTExGD48OGya2pUEc7JfYr7C3VfSUkJFi5ciODgYNy+fRsXL16Ek5MT5syZAwcHB/j4+IiOSOUUC1Iion+pon1xGjVqFLy9vUsVpH/99RfWrVuHI0eOiAmmATk5ObLc12RlZYWTJ0/igw8+gFKplFWR9rzff/8d+/fvR4cOHURHoTLw8vISHYHKaMGCBdi4cSMWLVqE//73v6rxJk2aYPny5SxI6aW4ZJeIiF7JzMwMZ86cgbOzs9r45cuX0aJFC2RlZYkJpgFVqlTBkCFD4O3tLauCZt68efD393+tQlTXz3ds0KABwsLC0LRpU9FRqIyKi4vxyy+/IDExEZIkwcXFBQMGDICenp7oaPQanJ2dsWbNGnTr1k3tHNKkpCS0bdsWDx48EB2RyinOkBIRaUheXh4KCwvVxszMzASl0RxJkvD333+XGn/48KHOFzNbt27Fhg0b0K1bN9jb28Pb2xsjR46Era2t6GhlMm/ePHh6euLy5csYMGAA1q9fj6pVq4qOpRVLly7FjBkzEBwczOXjOuzy5cvo27cvbty4gfr160OpVOLixYuoXbs2fv/9d9SpU0d0RPoHN27cKPXiEniylPf5341Ez+IMKRFRGeTk5GDmzJkICwtDZmZmqeu6XrABQP/+/VGpUiVs3bpVNVNRXFwMDw8P5OTkYO/evYITll1mZiZCQ0OxYcMGJCQkoFevXvD29saAAQOgr6/b727nz5+P6dOno1KlSqKjaAWb4chD3759oVQqsXnzZtVZ1ZmZmRg+fDgUCgV+//13wQnpn7Ro0QKffvophg8frjZDOn/+fBw6dAh//vmn6IhUTrEgJSIqgwkTJuDw4cPw9/fHyJEjsWrVKty4cQNr1qxBQEAAhg0bJjpimSUkJKBTp06oWrUqOnbsCAD4888/kZ2djcjISDRu3FhwQs369ttvMX36dBQUFKB69eoYN24c/Pz8ZFvQ6bqNGze+8jr3JuqGypUr4+TJk2jSpInaeHx8PNq3b49Hjx4JSkava/fu3RgxYgRmzZoFf39/zJ8/H8nJyQgNDcVvv/2GHj16iI5I5RQLUiKiMrCzs0NoaCg6d+6sttfyxx9/xNatW7Fnzx7RETXi5s2bCAoKQnx8PExMTNC0aVN88sknqpkMXZeRkYHQ0FCsX78e6enpGDRoEHx8fHDz5k0EBASgRo0aOHDggOiYr6158+aIiIiAhYUFXF1dX7mXVJePRSH5sLS0xG+//YZ27dqpjUdFReGDDz7gTHc5lpqaCkdHR0iShP379+Orr75CTEyM6hzSL774Aj179hQdk8ox3V6HREQk2P379+Ho6AjgyX7Rp1+aOnTogPHjx4uMplG2trb46quvRMfQuPDwcKxfvx779++Hi4sLJkyYgOHDh6vtt2zWrBlcXV3FhXwDbm5uMDIyAgAMHDhQbJi3SK77uCuC/v37Y+zYsfjhhx/QqlUrAE86eY8bNw4DBgwQnI5epW7duqqjpXr16oWQkBBcvnwZ77zzjuhopCNYkBIRlYGTkxPS0tJgb28PFxcXhIWFoVWrVti9e7dsmsisX78eVapUwYcffqg2/tNPPyE3N1enl0SOHj0anp6eiIqKQsuWLV94j5OTE2bPnv2Wk5XN3LlzX/hnOaoI+7grgsDAQHh5eaFt27aqfcCFhYVwc3OrcEdt6ZrnF1vu3bsXX3/9taA0pIu4ZJeIqAyWL18OPT09+Pr64vDhw+jXrx+Ki4tRVFSEZcuWYdKkSaIjlln9+vURHByMLl26qI0fPXoUY8eORXJysqBkZZebm8u9oTquIuzjrkguX76MxMREKJVKuLi4vLBrK5UvCoUCGRkZsLa2BgC1hkZEr4MFKRGRBqWnpyM6Ohp16tTBu+++KzqORhgbGyMpKanUkRppaWlo2LAh8vLyxAR7Q9nZ2a99r64u97SwsHitM0gB3e9CW1H2ccvRlClTXvveZcuWaTEJlYWenh4yMjJgZWUF4ElBevbsWdV2FqJ/wiW7REQaZGdnBzs7O9ExNMra2hpnz54tVZDGx8ejWrVqYkKVQdWqVf+xWFMqlZAkSWeXe1akJY4VZR+3HMXGxr7Wfa/7coXEUCqVGDVqlGrf+uPHjzFu3DhUrlxZ7b7w8HAR8UgHsCAlIvqXAgMDMXbsWBgbGyMwMPCV9/r6+r6lVNrj6ekJX19fmJqaolOnTgCeLNedNGkSPD09Baf79w4fPiw6gtbp8r7ef6si7OOWq4rwd7EieP7fm+HDhwtKQrqKS3aJiP4lR0dHREdHo1q1aq9ckiRJElJTU99iMu0oKCjAiBEj8NNPP0Ff/8l7zOLiYnh5eSE4OBiGhoaCE9LrkmMX2oqwj5uISM5YkBIR0Ws5ffo0rly5gkqVKqFJkyawt7cXHUkjsrKycOrUKdy5cwclJSVq10aOHCkolebItQvts2cfPkuO+7iJiOSMBSkRURn4+/tj2rRppTq15uXlYfHixfjiiy8EJdOMrKwszJ49G9u3b8eDBw8APGmY4+npiQULFuj8ksjdu3dj2LBhyMnJgampqVpxI0mSzjf8AeTbhVZPT0919iEAeHh4IDAwEDY2NoKTERHRv8GClIioDJ7/UvxUZmYmrK2tdXb2CXjSLKZt27a4ceMGhg0bhoYNG0KpVCIxMRFbtmxB7dq1cfz4cVhYWIiO+sbq1auHvn374quvvpLt8S9y7ULLoyaIiOSBTY2IiMrgaTfW58XHx8PS0lJAIs3x9/eHoaEhUlJSSs06+fv7o2fPnvD398fy5csFJSy7GzduwNfXV7bFKMAutEREVL4pRAcgItJFFhYWsLS0hCRJqFevHiwtLVX/mZubo0ePHhgyZIjomGXyyy+/YMmSJS9cAvnOO+9g0aJF2Llzp4BkmtOrVy9ER0eLjqFVT7vQAlB1oQWg811oJUkq9TKIx4MQEekeLtklInoDGzduhFKphLe3N1asWAFzc3PVNUNDQzg4OKBt27YCE5adkZERUlJSUKtWrRdev379OpydnfH48eO3nKxsdu3apfrz3bt34e/vj9GjR6NJkyYwMDBQu3fAgAFvO57GybULrUKhQJ8+fVRnH+7evRtdu3bl2YdERDqGBSkR0RsqKirCpk2b0L1795cWbbqsZs2a2L59Ozp06PDC63/++Sc8PT1x48aNt5ysbBSK11scJEmSTu8Bfhm5dKEdPXr0a923fv16LSchIqKyYEFKRFQGlSpVQmJiomyOQHmWj48PLl++jIMHD5Y6azQ/Px+9evVCnTp18MMPPwhKSERERLqOBSkRURl06dIFkyZNwsCBA0VH0bjr16+jRYsWMDIywoQJE9CgQQMAQEJCAlavXo38/HxER0ejdu3agpO+udDQUHh4eKiWfT5VUFCAbdu2yeIcUn9//1de1/WjiYiISLexICUiKoOffvoJfn5+mDx5Mt57771S+9eaNm0qKJlmXLlyBR9//DEOHDiAp78uJElCjx49EBQUBGdnZ8EJy0bOx/Y85erqqva5sLAQV65cgb6+PurUqYMzZ84ISkZERMSClIioTF60H1GSJNVxMHIoaADgwYMHuHTpEgDA2dlZ54+0eUqhUOD27duwsrJSG4+Pj0eXLl1UR6TITXZ2NkaNGoVBgwZhxIgRouMQEVEFxoKUiKgMrl69+srrctxbKgeurq6QJAnx8fFo1KgR9PX/dyx3cXExrly5gt69e6uOSJGj8+fPo3///qojYYiIiETQ/+dbiIjoZVhw6qane37j4uLQq1cvVKlSRXXt6bE9gwcPFpTu7cjKysLDhw9FxyAiogqOBSkRkQYkJCQgPT0dBQUFauNyOMdSjubOnQsAcHBwgIeHB4yNjQUn0p7AwEC1z0qlErdu3cKPP/6I3r17C0pFRET0BJfsEhGVQWpqKgYNGoRz586p9o4CT/aRApDNHlK5KygowJ07d1BSUqI2bmdnJyiR5jg6Oqp9VigUsLKyQteuXTFr1iyYmpoKSkZERMSClIioTD744APo6enh+++/h5OTE06dOoXMzExMnToVS5YsQceOHUVHpFe4dOkSvL29cfz4cbVxuTWlIiIiKq+4ZJeIqAxOnDiByMhIWFlZQaFQQKFQoEOHDvj666/h6+uL2NhY0RHpFUaNGgV9fX389ttvqFGjhmpmW26ysrJw+fJlGBoawtHRkbOiRERUbrAgJSIqg+LiYlVDnOrVq+PmzZuoX78+7O3tkZycLDgd/ZO4uDjExMSgQYMGoqNoRVpaGiZMmID9+/erlpPr6+vD3d0dK1asgI2NDQAgPz8fRkZGIqMSEVEFxYKUiKgMGjdujLNnz8LJyQmtW7fGokWLYGhoiLVr18LJyUl0PPoHLi4uuHfvnugYWnHt2jW0adMGBgYG+PLLL9GwYUMolUokJibiu+++Q5s2bRAbG4s//vgDiYmJmDlzpujIRERUAXEPKRFRGezfvx85OTlwd3dHamoq+vfvj6SkJFSrVg3bt29H165dRUekV4iMjMTnn3+Or776Ck2aNIGBgYHadTMzM0HJys7b2xspKSnYv39/qS7CeXl56N27N0pKShAdHY1t27bBzc1NUFIiIqrIWJASEWnY/fv3YWFhIdv9iHKiUCgAoNTPSg5NjWxtbREWFoYOHTq88Poff/yBzp07Y926dfD29n7L6YiIiJ5gQUpERBXW0aNHX3n9/ffff0tJNM/IyAgpKSmoVavWC69fv34dTk5Opc7OJSIiepu4h5SI6A287oxSSEiIlpNQWehywflPbG1tceHChZcWpOfPn4etre1bTkVERKSOBSkR0RvYsGED7O3t4erqCi400W1ZWVn44YcfkJiYCEmS4OLiAm9vb5ibm4uOViZubm6YPn06mjdvDisrK7Vrd+7cwcyZMzFw4EAx4YiIiP4/LtklInoDH3/8MbZt2wY7Ozt4e3tj+PDhsLS0FB2L/qXo6Gj06tULJiYmaNWqFZRKJaKjo5GXl4cDBw6gefPmoiO+sQcPHqB169bIyMjA8OHDVUfbJCQkYMuWLXjnnXdw8uRJ/n9LRERCsSAlInpD+fn5CA8PR0hICI4fP45+/frBx8cHPXv2ZEMjHdGxY0c4Ozvj+++/h77+k0VDRUVFGDNmDFJTU/HHH38ITlg2Dx48wGeffYbt27cjKysLAFC1alUMGTIECxcuRLVq1cQGJCKiCo8FKRGRBly9ehUbNmxAaGgoCgsLkZCQgCpVqoiORf/AxMQEsbGxqtnDpxISEtCiRQvk5uYKSqZZSqUSd+/eBQBYWVnxhQkREZUbCtEBiIjkQJIkSJIEpVKJkpIS0XHoNZmZmSE9Pb3U+LVr12BqaiogkXZIkgRra2tYW1uzGCUionKFBSkR0RvKz8/H1q1b0aNHD9SvXx/nzp1DUFAQ0tPTOTuqIzw8PODj44Pt27fj2rVruH79OrZt24YxY8bgo48+Eh2PiIhI9thll4joDTzb1Gj06NHYtm0b9+PpoCVLlkCSJIwcORJFRUUAAAMDA4wfPx4BAQGC0xEREckf95ASEb0BhUIBOzs7uLq6vnIJZHh4+FtMRW8qNzcXKSkpUCqVcHZ2RqVKlURHIiIiqhA4Q0pE9AZGjhzJvXgyUqlSJTRp0kR0DCIiogqHM6RERFTheHt7v9Z9ISEhWk6iHYGBga99r6+vrxaTEBERvRoLUiIiqnAUCgXs7e3h6uqKV/0a3Llz51tMpTmOjo6vdZ8kSUhNTdVyGiIiopdjQUpERBXOs02pvL29MXz4cFhaWoqORUREVOGwICUiogopPz8f4eHhCAkJwfHjx9GvXz/4+PigZ8+e3B9MRET0lrAgJSKiCu/q1avYsGEDQkNDUVhYiISEBFmdJXv9+nXs2rUL6enpKCgoULu2bNkyQamIiIjYZZeIiAiSJEGSJCiVSpSUlIiOo1EREREYMGAAHB0dkZycjMaNGyMtLQ1KpRLNmzcXHY+IiCo4hegAREREIuTn52Pr1q3o0aMH6tevj3PnziEoKAjp6emymh2dNWsWpk6divPnz8PY2Bg7duzAtWvX8P777+PDDz8UHY+IiCo4LtklIqIK59mmRqNHj8bw4cNRrVo10bG0wtTUFHFxcahTpw4sLCxw7NgxNGrUCPHx8XBzc0NaWproiEREVIFxyS4REVU4wcHBsLOzg6OjI44ePYqjR4++8L7w8PC3nEzzKleujPz8fACAra0tUlJS0KhRIwDAvXv3REYjIiJiQUpERBXPyJEjK0wn3TZt2iAqKgouLi7o168fpk6dinPnziE8PBxt2rQRHY+IiCo4LtklIiKSsdTUVDx69AhNmzZFbm4upk2bhmPHjsHZ2RnLly+Hvb296IhERFSBsSAlIiIiIiIiIbhkl4iIqAIoKCjAnTt3Sh1rY2dnJygRERERC1IiIiJZu3jxInx8fHD8+HG1caVSCUmSUFxcLCgZERERC1IiIiJZGz16NPT19fHbb7+hRo0aFaaZExER6QbuISUiIpKxypUrIyYmBg0aNBAdhYiIqBSF6ABERESkPS4uLjxvlIiIyi0WpERERDL2zTffYMaMGThy5AgyMzORnZ2t9h8REZFIXLJLREQkYwrFk3fPz+8dZVMjIiIqD9jUiIiISMYOHz4sOgIREdFLcYaUiIiIiIiIhOAMKRERkcxlZWXhhx9+QGJiIiRJgouLC7y9vWFubi46GhERVXCcISUiIpKx6Oho9OrVCyYmJmjVqhWUSiWio6ORl5eHAwcOoHnz5qIjEhFRBcaClIiISMY6duwIZ2dnfP/999DXf7IwqqioCGPGjEFqair++OMPwQmJiKgiY0FKREQkYyYmJoiNjUWDBg3UxhMSEtCiRQvk5uYKSkZERMRzSImIiGTNzMwM6enppcavXbsGU1NTAYmIiIj+hwUpERGRjHl4eMDHxwfbt2/HtWvXcP36dWzbtg1jxozBRx99JDoeERFVcOyyS0REJGNLliyBJEkYOXIkioqKAAAGBgYYP348AgICBKcjIqKKjntIiYiIKoDc3FykpKRAqVTC2dkZlSpVEh2JiIiIBSkRERERERGJwSW7REREMuPu7o4NGzbAzMwM7u7ur7w3PDz8LaUiIiIqjQUpERGRzJibm0OSJABPuuw+/TMREVF5wyW7REREREREJASPfSEiIpKxrl27Iisrq9R4dnY2unbt+vYDERERPYMzpERERDKmUCiQkZEBa2trtfE7d+6gZs2aKCwsFJSMiIiIe0iJiIhk6ezZs6o/JyQkICMjQ/W5uLgY+/btQ82aNUVEIyIiUuEMKRERkQwpFApVM6MX/ao3MTHBt99+C29v77cdjYiISIUFKRERkQxdvXoVSqUSTk5OOHXqFKysrFTXDA0NYW1tDT09PYEJiYiIWJASERERERGRINxDSkREVAEkJCQgPT0dBQUFauMDBgwQlIiIiIgFKRERkaylpqZi0KBBOHfuHCRJUu0nfbq/tLi4WGQ8IiKq4HgOKRERkYxNmjQJjo6OuH37NipVqoQLFy7gjz/+QIsWLXDkyBHR8YiIqILjHlIiIiIZq169OiIjI9G0aVOYm5vj1KlTqF+/PiIjIzF16lTExsaKjkhERBUYZ0iJiIhkrLi4GFWqVAHwpDi9efMmAMDe3h7JyckioxEREXEPKRERkZw1btwYZ8+ehZOTE1q3bo1FixbB0NAQa9euhZOTk+h4RERUwXHJLhERkYzt378fOTk5cHd3R2pqKvr374+kpCRUq1YN27dvR9euXUVHJCKiCowFKRERUQVz//59WFhYqDrtEhERicKClIiISMYePnyI4uJiWFpaqo3fv38f+vr6MDMzE5SMiIiITY2IiIhkzdPTE9u2bSs1HhYWBk9PTwGJiIiI/oczpERERDJmaWmJqKgoNGzYUG08KSkJ7du3R2ZmpqBkREREnCElIiKStfz8fBQVFZUaLywsRF5enoBERERE/8OClIiISMZatmyJtWvXlhoPDg7Ge++9JyARERHR//AcUiIiIhlbuHAhunfvjvj4eHTr1g0AEBERgdOnT+PAgQOC0xERUUXHPaREREQyFxcXh8WLFyMuLg4mJiZo2rQpZs2ahbp164qORkREFRwLUiIiIiIiIhKCS3aJiIhkLD09/ZXX7ezs3lISIiKi0jhDSkREJGMKhQKSJL30enFx8VtMQ0REpI4zpERERDIWGxur9rmwsBCxsbFYtmwZFi5cKCgVERHRE5whJSIiqoB+//13LF68GEeOHBEdhYiIKjCeQ0pERFQB1atXD6dPnxYdg4iIKjgu2SUiIpKx7Oxstc9KpRK3bt3CvHnzeOwLEREJx4KUiIhIxqpWrVqqqZFSqUTt2rWxbds2QamIiIie4B5SIiIiGTt69KjaZ4VCASsrKzg7O0Nfn++liYhILBakREREREREJARfjRIREcnMrl27XvveAQMGaDEJERHRq3GGlIiISGYUCvUm+pIk4dlf98/uKS0uLn5ruYiIiJ7HY1+IiIhkpqSkRPXfgQMH0KxZM+zduxdZWVl4+PAh9uzZg+bNm2Pfvn2ioxIRUQXHGVIiIiIZa9y4MYKDg9GhQwe18T///BNjx45FYmKioGREREScISUiIpK1lJQUmJublxo3NzdHWlra2w9ERET0DM6QEhERyVinTp1gYGCATZs2oUaNGgCAjIwMjBgxAgUFBaWOhSEiInqbWJASERHJ2OXLlzFo0CAkJyfDzs4OAJCeno569erhl19+gbOzs+CERERUkbEgJSIikjmlUomDBw8iKSkJSqUSLi4u6N69u1q3XSIiIhFYkBIREREREZEQbGpEREQkQ3379sXDhw9VnxcuXIisrCzV58zMTLi4uAhIRkRE9D+cISUiIpIhPT093Lp1C9bW1gAAMzMzxMXFwcnJCQBw+/Zt2Nraori4WGRMIiKq4DhDSkREJEPPv2/m+2ciIiqPWJASERERERGRECxIiYiIZEiSpFJddNlVl4iIyht90QGIiIhI85RKJUaNGgUjIyMAwOPHjzFu3DhUrlwZAJCfny8yHhEREQA2NSIiIpKl0aNHv9Z969ev13ISIiKil2NBSkREREREREJwDykREREREREJwYKUiIiIiIiIhGBBSkREREREREKwICUiIiIiIiIhWJASERERERGRECxIiYiIiIiISAgWpERERERERCQEC1IiIiIiIiISggUpERERERERCfH/AKsRGHA12TVwAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.heatmap(df.corr(),annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d7c28d0c-cc11-455b-ae0c-6a320e471da7",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop(columns=['Feedback'])\n",
    "y=df['Feedback']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "d924f3a6-c84d-4649-a10e-57406d0e462b",
   "metadata": {},
   "outputs": [],
   "source": [
    "vif=pd.DataFrame()\n",
    "vif['vif']=[variance_inflation_factor(x.values,i) for i in range((x.shape[1]))]\n",
    "vif['features_names']=x.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "a8bd1d73-3025-4969-8147-62e1282bf7d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>vif</th>\n",
       "      <th>features_names</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>149.328470</td>\n",
       "      <td>Age</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.420291</td>\n",
       "      <td>Gender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.119851</td>\n",
       "      <td>Martial stattus</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.598461</td>\n",
       "      <td>Occupation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.155710</td>\n",
       "      <td>Monthly Income</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2.256896</td>\n",
       "      <td>Educational Qualifications</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7.417243</td>\n",
       "      <td>Family size</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>84974.192036</td>\n",
       "      <td>latitude</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>84621.985903</td>\n",
       "      <td>longitude</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>5.232080</td>\n",
       "      <td>Output</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            vif              features_names\n",
       "0    149.328470                         Age\n",
       "1      2.420291                      Gender\n",
       "2      7.119851             Martial stattus\n",
       "3      7.598461                  Occupation\n",
       "4      7.155710              Monthly Income\n",
       "5      2.256896  Educational Qualifications\n",
       "6      7.417243                 Family size\n",
       "7  84974.192036                    latitude\n",
       "8  84621.985903                   longitude\n",
       "9      5.232080                      Output"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vif"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c0901ced-dc00-467c-9d75-37bdbcd09bd3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(388, 10)\n",
      "(388,)\n"
     ]
    }
   ],
   "source": [
    "print(x.shape)\n",
    "print(y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "288c2f55-76c7-42be-ae0c-21da6d494654",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " the y values percentage Feedback\n",
      "1    0.81701\n",
      "0    0.18299\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(f\" the y values percentage {y.value_counts()/len(df)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "fd4217fa-0d3c-4ef2-a43f-e8df3a0a849c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " this the percentage of the training random sample Feedback\n",
      "1    0.806452\n",
      "0    0.193548\n",
      "Name: count, dtype: float64\n",
      " this the percentage of the test random sample Feedback\n",
      "1    0.858974\n",
      "0    0.141026\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42)\n",
    "print(f\" this the percentage of the training random sample {y_train.value_counts()/len(y_train)}\")\n",
    "print(f\" this the percentage of the test random sample {y_test.value_counts()/len(y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "68a104c3-22bf-4917-afca-672a26913c98",
   "metadata": {},
   "outputs": [],
   "source": [
    "sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)\n",
    "sss.get_n_splits(x, y)\n",
    "for train_index,test_index in sss.split(df,df['Feedback']):\n",
    "         stat_train=df.loc[train_index]\n",
    "         stat_test=df.loc[test_index]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "eab6891a-6308-418e-ae9a-38d20a8b046a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the percentage of train straitifiedFeedback\n",
      "1    0.816129\n",
      "0    0.183871\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(f\"this is the percentage of train straitified{stat_train['Feedback'].value_counts()/len(stat_train)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "9de0e4b8-54d9-4a37-8e09-3cff634f3e87",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " this the percentage of test stra Feedback\n",
      "1    0.820513\n",
      "0    0.179487\n",
      "Name: count, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(f\" this the percentage of test stra {stat_test['Feedback'].value_counts()/len(stat_test)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "1be47571-f872-4003-bc62-ba25b5c37183",
   "metadata": {},
   "outputs": [],
   "source": [
    "stat_train_x=stat_train.drop(columns=['Feedback'])\n",
    "\n",
    "stat_train_y=stat_train['Feedback'] \n",
    "\n",
    "stat_test_x=stat_test.drop(columns=['Feedback'])\n",
    "\n",
    "stat_test_y=stat_test['Feedback'] \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "565f0825-712f-4cea-a19e-c72c2f44fc71",
   "metadata": {},
   "outputs": [],
   "source": [
    "scalar=StandardScaler() \n",
    "trains=scalar.fit_transform(stat_train_x)\n",
    "normal_trains=pd.DataFrame(trains,columns=stat_train_x.columns)\n",
    "normal_trains\n",
    "tests=scalar.fit_transform(stat_test_x)\n",
    "normal_tests=pd.DataFrame(tests,columns=stat_test_x.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "ca919a86-d39b-4a54-a1fb-e5195c88ba02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Martial stattus</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Monthly Income</th>\n",
       "      <th>Educational Qualifications</th>\n",
       "      <th>Family size</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>Output</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>310.000000</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "      <td>3.100000e+02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>4.297638e-17</td>\n",
       "      <td>8.308766e-17</td>\n",
       "      <td>8.595275e-17</td>\n",
       "      <td>2.148819e-17</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.031433e-16</td>\n",
       "      <td>-8.022257e-17</td>\n",
       "      <td>-4.884408e-14</td>\n",
       "      <td>-6.991970e-14</td>\n",
       "      <td>1.146037e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "      <td>1.001617e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-2.274888e+00</td>\n",
       "      <td>-1.208436e+00</td>\n",
       "      <td>-1.548593e+00</td>\n",
       "      <td>-1.416313e+00</td>\n",
       "      <td>-1.816173</td>\n",
       "      <td>-1.040471e+00</td>\n",
       "      <td>-1.713917e+00</td>\n",
       "      <td>-2.133899e+00</td>\n",
       "      <td>-2.314019e+00</td>\n",
       "      <td>-1.818119e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-5.590358e-01</td>\n",
       "      <td>-1.208436e+00</td>\n",
       "      <td>-1.548593e+00</td>\n",
       "      <td>-1.416313e+00</td>\n",
       "      <td>-1.142712</td>\n",
       "      <td>-1.040471e+00</td>\n",
       "      <td>-9.676891e-01</td>\n",
       "      <td>-6.993323e-01</td>\n",
       "      <td>-5.842418e-01</td>\n",
       "      <td>5.500191e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>-2.158653e-01</td>\n",
       "      <td>8.275159e-01</td>\n",
       "      <td>6.687919e-01</td>\n",
       "      <td>8.429857e-01</td>\n",
       "      <td>0.204211</td>\n",
       "      <td>-6.894689e-02</td>\n",
       "      <td>-2.214612e-01</td>\n",
       "      <td>8.920457e-02</td>\n",
       "      <td>-2.270893e-01</td>\n",
       "      <td>5.500191e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>4.704757e-01</td>\n",
       "      <td>8.275159e-01</td>\n",
       "      <td>6.687919e-01</td>\n",
       "      <td>8.429857e-01</td>\n",
       "      <td>0.877672</td>\n",
       "      <td>9.025774e-01</td>\n",
       "      <td>5.247667e-01</td>\n",
       "      <td>5.571014e-01</td>\n",
       "      <td>7.072991e-01</td>\n",
       "      <td>5.500191e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2.529499e+00</td>\n",
       "      <td>8.275159e-01</td>\n",
       "      <td>6.687919e-01</td>\n",
       "      <td>8.429857e-01</td>\n",
       "      <td>0.877672</td>\n",
       "      <td>2.845626e+00</td>\n",
       "      <td>2.017223e+00</td>\n",
       "      <td>3.058093e+00</td>\n",
       "      <td>2.975701e+00</td>\n",
       "      <td>5.500191e-01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Age        Gender  Martial stattus    Occupation  \\\n",
       "count  3.100000e+02  3.100000e+02     3.100000e+02  3.100000e+02   \n",
       "mean   4.297638e-17  8.308766e-17     8.595275e-17  2.148819e-17   \n",
       "std    1.001617e+00  1.001617e+00     1.001617e+00  1.001617e+00   \n",
       "min   -2.274888e+00 -1.208436e+00    -1.548593e+00 -1.416313e+00   \n",
       "25%   -5.590358e-01 -1.208436e+00    -1.548593e+00 -1.416313e+00   \n",
       "50%   -2.158653e-01  8.275159e-01     6.687919e-01  8.429857e-01   \n",
       "75%    4.704757e-01  8.275159e-01     6.687919e-01  8.429857e-01   \n",
       "max    2.529499e+00  8.275159e-01     6.687919e-01  8.429857e-01   \n",
       "\n",
       "       Monthly Income  Educational Qualifications   Family size      latitude  \\\n",
       "count      310.000000                3.100000e+02  3.100000e+02  3.100000e+02   \n",
       "mean         0.000000               -1.031433e-16 -8.022257e-17 -4.884408e-14   \n",
       "std          1.001617                1.001617e+00  1.001617e+00  1.001617e+00   \n",
       "min         -1.816173               -1.040471e+00 -1.713917e+00 -2.133899e+00   \n",
       "25%         -1.142712               -1.040471e+00 -9.676891e-01 -6.993323e-01   \n",
       "50%          0.204211               -6.894689e-02 -2.214612e-01  8.920457e-02   \n",
       "75%          0.877672                9.025774e-01  5.247667e-01  5.571014e-01   \n",
       "max          0.877672                2.845626e+00  2.017223e+00  3.058093e+00   \n",
       "\n",
       "          longitude        Output  \n",
       "count  3.100000e+02  3.100000e+02  \n",
       "mean  -6.991970e-14  1.146037e-16  \n",
       "std    1.001617e+00  1.001617e+00  \n",
       "min   -2.314019e+00 -1.818119e+00  \n",
       "25%   -5.842418e-01  5.500191e-01  \n",
       "50%   -2.270893e-01  5.500191e-01  \n",
       "75%    7.072991e-01  5.500191e-01  \n",
       "max    2.975701e+00  5.500191e-01  "
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "normal_trains.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b653bfb-7d92-43bf-a0d8-857db8e41eb6",
   "metadata": {},
   "source": [
    "Train the model using all features  basically using Log model without normalizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "2934e738-edd5-4924-baf1-64bec993babf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8461538461538461"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model=LogisticRegression() \n",
    "log_model.fit(x_train,y_train)\n",
    "pred_log=log_model.score(x_test,y_test)\n",
    "pred_log"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc42075c-3e2c-4ab9-9c61-509631c7f4e2",
   "metadata": {},
   "source": [
    "Train the model using all features basically using Log model without  normalizations and its stratified"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "d60ace77-ad71-490e-a103-26f79a749061",
   "metadata": {},
   "outputs": [],
   "source": [
    "st_x_train=stat_train.drop(columns=['Feedback'])\n",
    "st_y_train=stat_train['Feedback']\n",
    "st_x_test=stat_test.drop(columns=['Feedback'])\n",
    "st_y_test=stat_test['Feedback']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "85bb7ff5-57d6-4753-9a80-397f385a0460",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8717948717948718"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model_stra=LogisticRegression() \n",
    "log_model_stra.fit(st_x_train,st_y_train)\n",
    "pred_log_st=log_model_stra.score(st_x_test,st_y_test)\n",
    "pred_log_st"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69d35b02-392f-4625-a0e7-9083c7c9d02c",
   "metadata": {},
   "source": [
    "Train the model using all features basically using Log model with  normalizations and its stratified"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "02ec468d-18fb-4ec0-9943-d82aa9d0b779",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9102564102564102"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model_stra_n=LogisticRegression() \n",
    "log_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "pred_log_st=log_model_stra.score(normal_tests,stat_test_y)\n",
    "pred_log_st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "d9d880a3-b507-498b-a729-06a3657546a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>vif</th>\n",
       "      <th>features_names</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.182332</td>\n",
       "      <td>Age</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.047240</td>\n",
       "      <td>Gender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2.130963</td>\n",
       "      <td>Martial stattus</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.653918</td>\n",
       "      <td>Occupation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.646331</td>\n",
       "      <td>Monthly Income</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.093716</td>\n",
       "      <td>Educational Qualifications</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1.102011</td>\n",
       "      <td>Family size</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1.077037</td>\n",
       "      <td>latitude</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1.098677</td>\n",
       "      <td>longitude</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1.178743</td>\n",
       "      <td>Output</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        vif              features_names\n",
       "0  2.182332                         Age\n",
       "1  1.047240                      Gender\n",
       "2  2.130963             Martial stattus\n",
       "3  2.653918                  Occupation\n",
       "4  1.646331              Monthly Income\n",
       "5  1.093716  Educational Qualifications\n",
       "6  1.102011                 Family size\n",
       "7  1.077037                    latitude\n",
       "8  1.098677                   longitude\n",
       "9  1.178743                      Output"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vif_n=pd.DataFrame()\n",
    "vif_n['vif']=[variance_inflation_factor(normal_trains.values,i) for i in range((normal_trains.shape[1]))]\n",
    "vif_n['features_names']=normal_trains.columns\n",
    "vif_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "bb3556a0-af80-4d5d-abc3-2ec55624871e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Age', 'Gender', 'Martial stattus', 'Occupation', 'Monthly Income',\n",
       "       'Educational Qualifications', 'Family size', 'latitude',\n",
       "       'longitude', 'Output'], dtype=object)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model_stra_n.feature_names_in_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "1a03b6f5-bdc1-46b0-a528-4c0c766114c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>PCA()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;PCA<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html\">?<span>Documentation for PCA</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>PCA()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "PCA()"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "componet=None\n",
    "pca=PCA(n_components=componet)\n",
    "pca.fit(normal_trains)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "54165f46-ed47-4156-99d2-28f42b258e19",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 28.32039701  41.03440084  52.54066147  63.24060009  73.03519939\n",
      "  80.90565508  87.83613121  94.21969961  97.3462992  100.        ]\n"
     ]
    }
   ],
   "source": [
    "print(pca.explained_variance_ratio_.cumsum()*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "04b82a7c-4b8e-40fa-b425-8aff997a6c17",
   "metadata": {},
   "outputs": [],
   "source": [
    "componet=len(pca.explained_variance_ratio_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "9fad6ae6-c84a-4ebe-b456-fc90bf3d4f07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'explance_variance')"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(1,componet+1),np.cumsum(pca.explained_variance_ratio_*100))\n",
    "plt.xlabel('number of componet')\n",
    "plt.ylabel(\"explance_variance\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "37b6edfc-ab7a-45ad-8d9c-f20ecf2ff1a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-2 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-2 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-2 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-2 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-2 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>PCA(n_components=9)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;PCA<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html\">?<span>Documentation for PCA</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>PCA(n_components=9)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "PCA(n_components=9)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "componets=9\n",
    "pca_new=PCA(n_components=componets)\n",
    "pca_new.fit(normal_trains)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "01f2a15b-e509-464e-a1b7-5d2636928dd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[28.32039701 41.03440084 52.54066147 63.24060009 73.03519939 80.90565508\n",
      " 87.83613121 94.21969961 97.3462992 ]\n"
     ]
    }
   ],
   "source": [
    "print(pca_new.explained_variance_ratio_.cumsum()*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "d3c60f67-99e2-49ed-b2b2-b79e1129bfe0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.4710969  0.03306597 0.47909261 0.51493621 0.39176759 0.10955016\n",
      "  0.15261034 0.05552638 0.12791832 0.2721508 ]\n",
      " [0.15366159 0.01979403 0.1163181  0.06563686 0.08397374 0.359825\n",
      "  0.09087001 0.58952113 0.57960218 0.36041835]\n",
      " [0.221864   0.09296647 0.21761173 0.14262484 0.33560479 0.4703335\n",
      "  0.62814063 0.30681209 0.08060969 0.21307386]\n",
      " [0.15157182 0.8775751  0.04806546 0.04148273 0.00752997 0.05877185\n",
      "  0.0991758  0.22424702 0.33497544 0.16436204]\n",
      " [0.09550278 0.14661386 0.02824894 0.18557491 0.28271058 0.62644807\n",
      "  0.53247693 0.16976257 0.0224267  0.38591594]\n",
      " [0.18241324 0.35319791 0.06495731 0.00880955 0.07565556 0.20790524\n",
      "  0.11604913 0.58846075 0.62710225 0.18899878]\n",
      " [0.01368889 0.17491873 0.02819939 0.10295326 0.26630123 0.30957973\n",
      "  0.24427426 0.3468427  0.34797856 0.70000476]\n",
      " [0.31286762 0.1627065  0.40151407 0.02404929 0.61860912 0.28666373\n",
      "  0.4463161  0.03086141 0.05112395 0.21501409]\n",
      " [0.62926963 0.12331648 0.72518929 0.14142853 0.12368673 0.10669115\n",
      "  0.02661014 0.07384958 0.08664429 0.05060369]]\n"
     ]
    }
   ],
   "source": [
    "import_features_pca=abs(pca_new.components_) \n",
    "print(import_features_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "a2a9f4cb-eb9d-4090-bdf1-fea191eaf0de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(310, 9)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_pca=pca_new.transform(normal_trains)\n",
    "x_pca.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cf266bd-d68e-4414-950d-8a2b82f7dec2",
   "metadata": {},
   "source": [
    "Creating a machine learning "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "470da67d-bb66-4cbb-ad5d-fe07d2c1c25d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc=StandardScaler()\n",
    "_pca=PCA(n_components=8) \n",
    "model=LogisticRegression() \n",
    "log_reg_pca=Pipeline([ ('scalar',sc) ,('pca',_pca),('models',model)])\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "78e1b2ad-abb2-432b-a1c6-2c7459445ab7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-3 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-3 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-3 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-3 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-3 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-3 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-3 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-3 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-3 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-3 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-3 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-3 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-3 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler()), (&#x27;pca&#x27;, PCA(n_components=8)),\n",
       "                (&#x27;models&#x27;, LogisticRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;Pipeline<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.pipeline.Pipeline.html\">?<span>Documentation for Pipeline</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler()), (&#x27;pca&#x27;, PCA(n_components=8)),\n",
       "                (&#x27;models&#x27;, LogisticRegression())])</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;StandardScaler<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html\">?<span>Documentation for StandardScaler</span></a></label><div class=\"sk-toggleable__content fitted\"><pre>StandardScaler()</pre></div> </div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;PCA<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html\">?<span>Documentation for PCA</span></a></label><div class=\"sk-toggleable__content fitted\"><pre>PCA(n_components=8)</pre></div> </div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression()</pre></div> </div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('scalar', StandardScaler()), ('pca', PCA(n_components=8)),\n",
       "                ('models', LogisticRegression())])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_trains,x_tests,y_trains,y_tests=train_test_split(x,y,test_size=0.2,shuffle=True)  \n",
    "# Training with pca component \n",
    "log_reg_pca.fit(x_trains,y_trains)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "4d08bf7b-5355-4b29-883e-d95ceb618c5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8589743589743589"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_reg_pca.score(x_tests,y_tests)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bab26ded-4967-40b4-9a42-d906d79b0558",
   "metadata": {},
   "source": [
    "Evaluating models only on the Training set "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "99d0c442-df5f-471f-801f-9bdb0e9974a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 42,  30],\n",
       "       [ 15, 223]])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model_stra_n=LogisticRegression() \n",
    "log_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "predict_log=log_model_stra.predict(normal_trains)\n",
    "log_conf=confusion_matrix(predict_log,stat_train_y)\n",
    "log_conf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "c2f82f60-2d7e-49e7-b423-7d08b4780b0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 56,   3],\n",
       "       [  1, 250]])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dec_model_stra_n=DecisionTreeClassifier() \n",
    "dec_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "predict_dec=dec_model_stra_n.predict(normal_trains)\n",
    "dec_conf=confusion_matrix(predict_dec,stat_train_y)\n",
    "dec_conf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "6e92cf82-6007-41d2-98bc-0c4c53d5d8a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 56,   3],\n",
       "       [  1, 250]])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ran_model_stra_n=RandomForestClassifier() \n",
    "ran_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "predict_ran=ran_model_stra_n.predict(normal_trains)\n",
    "ran_conf=confusion_matrix(predict_ran,stat_train_y)\n",
    "ran_conf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "8f91e1d0-ee8c-47fb-916d-27eec3b1983d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 42,  13],\n",
       "       [ 15, 240]])"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ada_model_stra_n=AdaBoostClassifier() \n",
    "ada_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "predict_ada=ada_model_stra_n.predict(normal_trains)\n",
    "ada_conf=confusion_matrix(predict_ada,stat_train_y)\n",
    "ada_conf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "fb9603f1-f41c-4533-8374-03bf3df224f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 56,   3],\n",
       "       [  1, 250]])"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xg_model_stra_n=XGBClassifier()\n",
    "xg_model_stra_n.fit(normal_trains,stat_train_y)\n",
    "predict_xg=xg_model_stra_n.predict(normal_trains)\n",
    "xg_conf=confusion_matrix(predict_xg,stat_train_y)\n",
    "xg_conf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "c2a441d4-c61c-4025-be8f-2cde0fbfe365",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.74      0.58      0.65        72\n",
      "           1       0.88      0.94      0.91       238\n",
      "\n",
      "    accuracy                           0.85       310\n",
      "   macro avg       0.81      0.76      0.78       310\n",
      "weighted avg       0.85      0.85      0.85       310\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predict_log,stat_train_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "f4522d15-65d5-4120-880b-af5ad92b6d9c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.95      0.97        59\n",
      "           1       0.99      1.00      0.99       251\n",
      "\n",
      "    accuracy                           0.99       310\n",
      "   macro avg       0.99      0.97      0.98       310\n",
      "weighted avg       0.99      0.99      0.99       310\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predict_dec,stat_train_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "b57c413c-7531-49cf-9987-7498a386a8be",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.95      0.97        59\n",
      "           1       0.99      1.00      0.99       251\n",
      "\n",
      "    accuracy                           0.99       310\n",
      "   macro avg       0.99      0.97      0.98       310\n",
      "weighted avg       0.99      0.99      0.99       310\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predict_ran,stat_train_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "e98ca7a1-fe75-49f6-84f9-035d89baea15",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.74      0.76      0.75        55\n",
      "           1       0.95      0.94      0.94       255\n",
      "\n",
      "    accuracy                           0.91       310\n",
      "   macro avg       0.84      0.85      0.85       310\n",
      "weighted avg       0.91      0.91      0.91       310\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predict_ada,stat_train_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "cd14b12c-6273-4740-ad98-bc00c11c7483",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.95      0.97        59\n",
      "           1       0.99      1.00      0.99       251\n",
      "\n",
      "    accuracy                           0.99       310\n",
      "   macro avg       0.99      0.97      0.98       310\n",
      "weighted avg       0.99      0.99      0.99       310\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predict_xg,stat_train_y))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be25a5f8-8b20-48a7-9c31-797ee3589061",
   "metadata": {},
   "source": [
    "check if the models overfited "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "a5044c86-8995-4dda-8f66-08ed623c8b4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the cross validationset of log model on the training set:0.8193925815285038\n"
     ]
    }
   ],
   "source": [
    "cv_log=cross_val_score(log_model_stra_n,normal_trains,stat_train_y,cv=3,scoring='accuracy')\n",
    "print(f\"this is the cross validationset of log model on the training set:{cv_log.mean()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "5094e9a9-ae68-4b0a-827a-ebf68163813d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the cross validationset of dec model on the training set:0.8193303460293752\n"
     ]
    }
   ],
   "source": [
    "cv_dec=cross_val_score(dec_model_stra_n,normal_trains,stat_train_y,cv=3,scoring='accuracy')\n",
    "print(f\"this is the cross validationset of dec model on the training set:{cv_dec.mean()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "9751aaf7-37ee-47fb-ab4c-8876ef4c374e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the cross validationset of ran model on the training set:0.8451580781677869\n"
     ]
    }
   ],
   "source": [
    "cv_ran=cross_val_score(ran_model_stra_n,normal_trains,stat_train_y,cv=3,scoring='accuracy')\n",
    "print(f\"this is the cross validationset of ran model on the training set:{cv_ran.mean()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "bb029acc-5b45-4d18-a732-32119f569af5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the cross validationset of ada model on the training set:0.832244212098581\n"
     ]
    }
   ],
   "source": [
    "cv_ada=cross_val_score(ada_model_stra_n,normal_trains,stat_train_y,cv=3,scoring='accuracy')\n",
    "print(f\"this is the cross validationset of ada model on the training set:{cv_ada.mean()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "c4541e20-7f03-44e7-981a-a44144a61d45",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is the cross validationset of ada model on the training set:0.8483320886233509\n"
     ]
    }
   ],
   "source": [
    "cv_xg=cross_val_score(xg_model_stra_n,normal_trains,stat_train_y,cv=3,scoring='accuracy')\n",
    "print(f\"this is the cross validationset of ada model on the training set:{cv_xg.mean()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "52c64323-caf2-4288-925a-9d24cca9ac9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the classificationreport on tes data               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.79      0.92      0.85        12\n",
      "           1       0.98      0.95      0.97        66\n",
      "\n",
      "    accuracy                           0.95        78\n",
      "   macro avg       0.89      0.94      0.91        78\n",
      "weighted avg       0.95      0.95      0.95        78\n",
      "\n"
     ]
    }
   ],
   "source": [
    "log_final=LogisticRegression() \n",
    "log_final.fit(normal_tests,stat_test_y)\n",
    "predict_log_test=log_final.predict(normal_tests)\n",
    "log_conft=confusion_matrix(predict_log_test,stat_test_y)\n",
    "\n",
    "print(f\"the classificationreport on tes data {classification_report(predict_log_test,stat_test_y)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "40c8df30-d66d-47c0-a15a-66e285b71e8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the classificationreport on dec data               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        14\n",
      "           1       1.00      1.00      1.00        64\n",
      "\n",
      "    accuracy                           1.00        78\n",
      "   macro avg       1.00      1.00      1.00        78\n",
      "weighted avg       1.00      1.00      1.00        78\n",
      "\n"
     ]
    }
   ],
   "source": [
    "dec_final=DecisionTreeClassifier() \n",
    "dec_final.fit(normal_tests,stat_test_y)\n",
    "predict_dec_test=dec_final.predict(normal_tests)\n",
    "dec_conft=confusion_matrix(predict_dec_test,stat_test_y)\n",
    "\n",
    "print(f\"the classificationreport on dec data {classification_report(predict_dec_test,stat_test_y)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "82f0f2ff-551b-4c50-96f6-97a9ca15fd69",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the classificationreport on random data               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        14\n",
      "           1       1.00      1.00      1.00        64\n",
      "\n",
      "    accuracy                           1.00        78\n",
      "   macro avg       1.00      1.00      1.00        78\n",
      "weighted avg       1.00      1.00      1.00        78\n",
      "\n"
     ]
    }
   ],
   "source": [
    "ran_final=RandomForestClassifier() \n",
    "ran_final.fit(normal_tests,stat_test_y)\n",
    "predict_ran_test=ran_final.predict(normal_tests)\n",
    "ran_conft=confusion_matrix(predict_ran_test,stat_test_y)\n",
    "\n",
    "print(f\"the classificationreport on random data {classification_report(predict_ran_test,stat_test_y)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "fbad1b13-709a-45d5-89c9-2697e424a6ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the classificationreport on tes data               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.79      0.92      0.85        12\n",
      "           1       0.98      0.95      0.97        66\n",
      "\n",
      "    accuracy                           0.95        78\n",
      "   macro avg       0.89      0.94      0.91        78\n",
      "weighted avg       0.95      0.95      0.95        78\n",
      "\n"
     ]
    }
   ],
   "source": [
    "log_final=LogisticRegression() \n",
    "log_final.fit(normal_tests,stat_test_y)\n",
    "predict_log_test=log_final.predict(normal_tests)\n",
    "log_conft=confusion_matrix(predict_log_test,stat_test_y)\n",
    "\n",
    "print(f\"the classificationreport on tes data {classification_report(predict_log_test,stat_test_y)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
