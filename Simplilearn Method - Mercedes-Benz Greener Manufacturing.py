{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Import and Load the Training and Testing Data into Dataframes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read the test and Train Data and load into DataFrames\n",
    "\n",
    "Trainpath = 'C:/Users/Manasvi/Documents/Kalpesh Data/Final Project 1 - Mercedes Benz/Final Project 1 - Mercedes Benz/train.csv'\n",
    "MercedesTrain = pd.read_csv(Trainpath)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 378 entries, ID to X385\n",
      "dtypes: float64(1), int64(369), object(8)\n",
      "memory usage: 12.1+ MB\n"
     ]
    }
   ],
   "source": [
    "# Displaying the information of Tran Dataset for overall look\n",
    "MercedesTrain.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "Testpath = 'C:/Users/Manasvi/Documents/Kalpesh Data/Final Project 1 - Mercedes Benz/Final Project 1 - Mercedes Benz/test.csv'\n",
    "MercedesTest = pd.read_csv(Testpath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 377 entries, ID to X385\n",
      "dtypes: int64(369), object(8)\n",
      "memory usage: 12.1+ MB\n"
     ]
    }
   ],
   "source": [
    "MercedesTest.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Drop the columns with zero variance from Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['X11' 'X93' 'X107' 'X233' 'X235' 'X268' 'X289' 'X290' 'X293' 'X297'\n",
      " 'X330' 'X347']\n"
     ]
    }
   ],
   "source": [
    "# Check the varience of the columns and lets Drop them. As their value does not affect the target.\n",
    "col_with_zero_var = MercedesTrain.var()[MercedesTrain.var() == 0 ].index.values\n",
    "print(col_with_zero_var)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "MercedesTrain.drop(col_with_zero_var,axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "MercedesTest.drop(col_with_zero_var,axis=1,inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check for Unique Values and Null values in Test and Train data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "M_train = np.array(MercedesTrain.isnull().sum())\n",
    "\n",
    "for i in range(0,len(M_train)):\n",
    "    if M_train[i] > 0:\n",
    "        print('The column {} is having null values'.format(i))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "M_test = np.array(MercedesTest.isnull().sum())\n",
    "\n",
    "for i in range (0,len(M_test)):\n",
    "    if M_test[i] > 0:\n",
    "        print('The column {} is having null values'.format(i))\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "0\n"
     ]
    }
   ],
   "source": [
    "# Alternate Method to check the Nulls inside the data\n",
    "\n",
    "print(np.sum(MercedesTrain.isnull().sum()))\n",
    "print(np.sum(MercedesTest.isnull().sum()))\n",
    "\n",
    "# As the sum is zero we conclude there are no missing values in Test and Train data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 366 entries, ID to X385\n",
      "dtypes: float64(1), int64(357), object(8)\n",
      "memory usage: 11.8+ MB\n"
     ]
    }
   ],
   "source": [
    "MercedesTrain.info() #Have a look before Dropping columns from Training DF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 365 entries, ID to X385\n",
      "dtypes: int64(357), object(8)\n",
      "memory usage: 11.7+ MB\n"
     ]
    }
   ],
   "source": [
    "MercedesTest.info() ## Have a look before Dropping columns from Testing DF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop the column name ID from both Test and Train Dataset\n",
    "MercedesTrain.drop('ID',axis=1,inplace = True)\n",
    "MercedesTest.drop('ID',axis=1,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 365 entries, y to X385\n",
      "dtypes: float64(1), int64(356), object(8)\n",
      "memory usage: 11.7+ MB\n"
     ]
    }
   ],
   "source": [
    "MercedesTrain.info() # Have a look after Dropping columns from Training DF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4209 entries, 0 to 4208\n",
      "Columns: 364 entries, X0 to X385\n",
      "dtypes: int64(356), object(8)\n",
      "memory usage: 11.7+ MB\n"
     ]
    }
   ],
   "source": [
    "MercedesTest.info()  # Have a look after Dropping columns from Testing DF"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As per above code output we conclude that there are no missing values in any of the columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Applying the Label encoder to the columns with data type Object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['X0' 'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X8']\n",
      "['X0' 'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X8']\n"
     ]
    }
   ],
   "source": [
    "# lets find out Object type columns from tets and tran datasets \n",
    "\n",
    "Obj_typ_col_train = MercedesTrain.describe(include=['object']).columns.values\n",
    "print(Obj_typ_col_train)\n",
    "\n",
    "Obj_typ_col_test = MercedesTest.describe(include=['object']).columns.values\n",
    "print(Obj_typ_col_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply Lable encoding to Object type columns.\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "for col in Obj_typ_col_train:\n",
    "    le.fit(MercedesTrain[col].append(MercedesTest[col]).values)\n",
    "    MercedesTrain[col] = le.transform(MercedesTrain[col])\n",
    "    MercedesTest[col] = le.transform(MercedesTest[col])\n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
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
       "      <th>y</th>\n",
       "      <th>X0</th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>X5</th>\n",
       "      <th>X6</th>\n",
       "      <th>X8</th>\n",
       "      <th>X10</th>\n",
       "      <th>...</th>\n",
       "      <th>X375</th>\n",
       "      <th>X376</th>\n",
       "      <th>X377</th>\n",
       "      <th>X378</th>\n",
       "      <th>X379</th>\n",
       "      <th>X380</th>\n",
       "      <th>X382</th>\n",
       "      <th>X383</th>\n",
       "      <th>X384</th>\n",
       "      <th>X385</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>130.81</td>\n",
       "      <td>37</td>\n",
       "      <td>23</td>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>27</td>\n",
       "      <td>9</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>88.53</td>\n",
       "      <td>37</td>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>31</td>\n",
       "      <td>11</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>76.26</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>38</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>30</td>\n",
       "      <td>9</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>80.62</td>\n",
       "      <td>24</td>\n",
       "      <td>21</td>\n",
       "      <td>38</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>30</td>\n",
       "      <td>11</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>78.02</td>\n",
       "      <td>24</td>\n",
       "      <td>23</td>\n",
       "      <td>38</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>14</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 365 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        y  X0  X1  X2  X3  X4  X5  X6  X8  X10  ...  X375  X376  X377  X378  \\\n",
       "0  130.81  37  23  20   0   3  27   9  14    0  ...     0     0     1     0   \n",
       "1   88.53  37  21  22   4   3  31  11  14    0  ...     1     0     0     0   \n",
       "2   76.26  24  24  38   2   3  30   9  23    0  ...     0     0     0     0   \n",
       "3   80.62  24  21  38   5   3  30  11   4    0  ...     0     0     0     0   \n",
       "4   78.02  24  23  38   5   3  14   3  13    0  ...     0     0     0     0   \n",
       "\n",
       "   X379  X380  X382  X383  X384  X385  \n",
       "0     0     0     0     0     0     0  \n",
       "1     0     0     0     0     0     0  \n",
       "2     0     0     1     0     0     0  \n",
       "3     0     0     0     0     0     0  \n",
       "4     0     0     0     0     0     0  \n",
       "\n",
       "[5 rows x 365 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# let us have a look into the data.\n",
    "MercedesTrain.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Perform the Dimentionallity Reduction using PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
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
       "      <th>X0</th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>X5</th>\n",
       "      <th>X6</th>\n",
       "      <th>X8</th>\n",
       "      <th>X10</th>\n",
       "      <th>X12</th>\n",
       "      <th>...</th>\n",
       "      <th>X375</th>\n",
       "      <th>X376</th>\n",
       "      <th>X377</th>\n",
       "      <th>X378</th>\n",
       "      <th>X379</th>\n",
       "      <th>X380</th>\n",
       "      <th>X382</th>\n",
       "      <th>X383</th>\n",
       "      <th>X384</th>\n",
       "      <th>X385</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>37</td>\n",
       "      <td>23</td>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>27</td>\n",
       "      <td>9</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>37</td>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>31</td>\n",
       "      <td>11</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>38</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>30</td>\n",
       "      <td>9</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>24</td>\n",
       "      <td>21</td>\n",
       "      <td>38</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>30</td>\n",
       "      <td>11</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>24</td>\n",
       "      <td>23</td>\n",
       "      <td>38</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>14</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4204</td>\n",
       "      <td>10</td>\n",
       "      <td>20</td>\n",
       "      <td>19</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4205</td>\n",
       "      <td>36</td>\n",
       "      <td>16</td>\n",
       "      <td>44</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4206</td>\n",
       "      <td>10</td>\n",
       "      <td>23</td>\n",
       "      <td>42</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4207</td>\n",
       "      <td>11</td>\n",
       "      <td>19</td>\n",
       "      <td>29</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>11</td>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4208</td>\n",
       "      <td>52</td>\n",
       "      <td>19</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>4209 rows × 364 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      X0  X1  X2  X3  X4  X5  X6  X8  X10  X12  ...  X375  X376  X377  X378  \\\n",
       "0     37  23  20   0   3  27   9  14    0    0  ...     0     0     1     0   \n",
       "1     37  21  22   4   3  31  11  14    0    0  ...     1     0     0     0   \n",
       "2     24  24  38   2   3  30   9  23    0    0  ...     0     0     0     0   \n",
       "3     24  21  38   5   3  30  11   4    0    0  ...     0     0     0     0   \n",
       "4     24  23  38   5   3  14   3  13    0    0  ...     0     0     0     0   \n",
       "...   ..  ..  ..  ..  ..  ..  ..  ..  ...  ...  ...   ...   ...   ...   ...   \n",
       "4204  10  20  19   2   3   1   3  16    0    0  ...     1     0     0     0   \n",
       "4205  36  16  44   3   3   1   7   7    0    0  ...     0     1     0     0   \n",
       "4206  10  23  42   0   3   1   6   4    0    1  ...     0     0     1     0   \n",
       "4207  11  19  29   5   3   1  11  20    0    0  ...     0     0     0     0   \n",
       "4208  52  19   5   2   3   1   6  22    0    0  ...     1     0     0     0   \n",
       "\n",
       "      X379  X380  X382  X383  X384  X385  \n",
       "0        0     0     0     0     0     0  \n",
       "1        0     0     0     0     0     0  \n",
       "2        0     0     1     0     0     0  \n",
       "3        0     0     0     0     0     0  \n",
       "4        0     0     0     0     0     0  \n",
       "...    ...   ...   ...   ...   ...   ...  \n",
       "4204     0     0     0     0     0     0  \n",
       "4205     0     0     0     0     0     0  \n",
       "4206     0     0     0     0     0     0  \n",
       "4207     0     0     0     0     0     0  \n",
       "4208     0     0     0     0     0     0  \n",
       "\n",
       "[4209 rows x 364 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Seperate out labels and Features\n",
    "\n",
    "Labels = MercedesTrain['y']\n",
    "\n",
    "Features = MercedesTrain\n",
    "Features.drop(['y'],axis=1,inplace=True)\n",
    "Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12\n",
      "[0.40868988 0.21758508 0.13120081 0.10783522 0.08165248 0.0140934\n",
      " 0.00660951 0.00384659 0.00260289 0.00214378 0.00209857 0.00180388]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "\n",
    "\n",
    "pca = PCA(0.98,svd_solver='full')\n",
    "pca.fit(Features)\n",
    "print(pca.n_components_)\n",
    "print(pca.explained_variance_ratio_)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Observations.\n",
    "\n",
    "1. From explained variance ration columns we can see that there are 12 columns which are having around 98% varience. So we will consider top 5 . If we add them all we will get around 92% varience."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "pca_final= PCA(n_components=5)\n",
    "pca_final=pca.fit(Features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(Features,Labels,test_size=0.2,random_state = 1)\n",
    "\n",
    "#X_train      : Train Data \n",
    "#X_test       : Validation Data\n",
    "#MercedesTest : Test Data \n",
    "\n",
    "pca_X_test = pd.DataFrame(pca_final.transform(X_test))\n",
    "pca_X_train = pd.DataFrame(pca_final.transform(X_train))\n",
    "pca_Mercedes_Test = pd.DataFrame(pca_final.transform(MercedesTest))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Applying Xgboost Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, gamma=0,\n",
       "             importance_type='gain', learning_rate=0.2, max_delta_step=0,\n",
       "             max_depth=4, min_child_weight=1, missing=None, n_estimators=100,\n",
       "             n_jobs=1, nthread=None, objective='reg:squarederror',\n",
       "             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,\n",
       "             seed=None, silent=None, subsample=1, verbosity=1)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import xgboost\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "model = XGBRegressor(objective='reg:squarederror',learning_rate=0.2,max_depth=4)\n",
    "model.fit(pca_X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score 0.7453547382059571 RMSE Score 40.75939209118279\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score,mean_squared_error\n",
    "\n",
    "# Calculating predicated value of y usng X_train values.\n",
    "XGB_Y_pred = model.predict(pca_X_train)\n",
    "\n",
    "# Calculating R2 Score for the model\n",
    "XGB_r2Score = r2_score(y_train,XGB_Y_pred)\n",
    "\n",
    "# Calculating RMSE score of the model.\n",
    "XGB_RMSE_Score = mean_squared_error(y_train,XGB_Y_pred)\n",
    "\n",
    "# Finally printing them all together\n",
    "\n",
    "print ('R2 Score {} RMSE Score {}'.\n",
    "       format(XGB_r2Score,XGB_RMSE_Score))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Use RamdomForestRegressor as alternate algo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, gamma=0,\n",
       "             importance_type='gain', learning_rate=0.2, max_delta_step=0,\n",
       "             max_depth=4, min_child_weight=1, missing=None, n_estimators=100,\n",
       "             n_jobs=1, nthread=None, objective='reg:squarederror',\n",
       "             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,\n",
       "             seed=None, silent=None, subsample=1, verbosity=1)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating model with RandomForestRegressor and compare the models.\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import xgboost\n",
    "from xgboost import XGBRFRegressor\n",
    "\n",
    "modelRF = XGBRegressor(objective='reg:squarederror',learning_rate=0.2,max_depth=4)\n",
    "modelRF.fit(pca_X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score 0.7453547382059571 RMSE Score 40.75939209118279\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score,mean_squared_error\n",
    "\n",
    "\n",
    "# Calculating predicated value of y usng X_train values.\n",
    "XGB_RF__Y_pred = modelRF.predict(pca_X_train)\n",
    "\n",
    "# Calculating R2 Score for the model\n",
    "XGB_RF_r2Score = r2_score(y_train,XGB_RF__Y_pred)\n",
    "\n",
    "# Calculating RMSE score of the model.\n",
    "XGB_RF_RMSE_Score = mean_squared_error(y_train,XGB_RF__Y_pred)\n",
    "\n",
    "# Finally printing them all together\n",
    "\n",
    "print ('R2 Score {} RMSE Score {}'.\n",
    "       format(XGB_RF_r2Score,XGB_RF_RMSE_Score))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Observations and Conclusion : From RMSE and R2 score we conclude that both the models are performing same"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predict your test_df values using XGBoost."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using Xgboost Regressor\n",
    "XGB_Y_pred = model.predict(pca_Mercedes_Test)\n",
    "# Using Ramdom Forest Regressor\n",
    "XGB_RF__Y_pred = modelRF.predict(pca_Mercedes_Test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
