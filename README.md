
# Preprocessing Text Data with Encoders

This project focuses on preprocessing a text-based dataset using various encoding techniques to transform categorical variables into numeric format suitable for machine learning models.

## Dataset Overview

The dataset contains the following features:

- `backpack_color`: The color of the student's backpack (categorical).
- `grade`: The grade received by the student (ordinal categorical).
- `favorite_creature`: The student's favorite mythical creature (categorical).
- `arrived`: The target variable indicating whether the student arrived on time (`on time`) or late (`late`).

### Initial Data Preview

The dataset is loaded and a brief preview is shown:

```python
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/text-data.csv')
df.head()
```

## Data Preprocessing

### Splitting the Data

The dataset is split into features (`X`) and the target variable (`y`), and then further split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='arrived')
y = df['arrived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
```

### Encoding Categorical Variables

Different encoding techniques are used based on the nature of the categorical variables.

#### `backpack_color` Encoding

The `backpack_color` column is encoded using OneHotEncoder, dropping the first category to avoid multicollinearity:

```python
from sklearn.preprocessing import OneHotEncoder

backpack_color_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
backpack_color_ohe.fit(X_train['backpack_color'].values.reshape(-1,1))
```

#### `grade` Encoding

The `grade` column, being an ordinal variable, is encoded using OrdinalEncoder:

```python
from sklearn.preprocessing import OrdinalEncoder

grade_ord_enc = OrdinalEncoder(categories=[['F', 'D', 'C', 'B', 'A']], encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
grade_ord_enc.fit(X_train['grade'].values.reshape(-1,1))
```

#### `favorite_creature` Encoding

The `favorite_creature` column is encoded using OneHotEncoder, treating rare categories as infrequent:

```python
creature_ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, min_frequency=0.2)
creature_ohe.fit(X_train['favorite_creature'].values.reshape(-1,1))
```

## Preprocessing Function

A preprocessing function is created to apply the trained encoders to new data, including the testing set:

```python
def X_preprocess(X_data):
    # Transform each column into numpy arrays
    backpack_color_encoded = backpack_color_ohe.transform(X_data['backpack_color'].values.reshape(-1,1))
    grade_encoded = grade_ord_enc.transform(X_data['grade'].values.reshape(-1,1))
    favorite_creature_encoded = creature_ohe.transform(X_data['favorite_creature'].values.reshape(-1,1))

    # Reorganize the numpy arrays into a DataFrame
    backpack_color_df = pd.DataFrame(backpack_color_encoded, columns=backpack_color_ohe.get_feature_names_out())
    creature_df = pd.DataFrame(favorite_creature_encoded, columns=creature_ohe.get_feature_names_out())
    out_df = pd.concat([backpack_color_df, creature_df], axis=1)
    out_df['grade'] = grade_encoded

    # Return the DataFrame
    return out_df
```

### Applying the Preprocessing Function

The preprocessing function is applied to both the training and testing sets:

```python
X_train_clean = X_preprocess(X_train)
X_test_clean = X_preprocess(X_test)
```

### Resulting DataFrames

After preprocessing, the categorical variables are transformed into a numeric format:

- For `backpack_color`, the resulting DataFrame includes binary columns for each color (except the dropped category).
- For `grade`, the values are ordinal, with higher grades receiving higher numerical values.
- For `favorite_creature`, rare categories are grouped under an "infrequent" label.

Example of preprocessed data:

```python
X_train_clean.head()
```

| x0_red | x0_yellow | x0_dragon | x0_griffin | x0_infrequent_sklearn | grade |
|--------|-----------|-----------|------------|-----------------------|-------|
| 1.0    | 0.0       | 0.0       | 1.0        | 0.0                   | 3.0   |
| 0.0    | 0.0       | 1.0       | 0.0        | 0.0                   | 1.0   |
| 0.0    | 0.0       | 0.0       | 0.0        | 1.0                   | 1.0   |
| 1.0    | 0.0       | 0.0       | 0.0        | 1.0                   | 0.0   |
| 0.0    | 0.0       | 0.0       | 1.0        | 0.0                   | 2.0   |

## Conclusion

This project successfully demonstrates how to preprocess categorical text data using various encoding techniques. The resulting numeric data is now ready for use in machine learning models, ensuring that the categorical variables are appropriately transformed for effective modeling.
```

This README provides a clear and comprehensive overview of the preprocessing steps taken for the categorical text data, including encoding techniques and the resulting transformations.
