from utils import db_connect
engine = db_connect()

# your code here
from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split

# Carga de datos
df = pd.read_csv("../data/raw/AB_NYC_2019.csv")

# Eliminación de columnas irrelevantes
df = df.drop(['id', 'name', 'host_name', 'host_id', 'last_review', 'longitude', 'latitude', 'neighbourhood'], axis=1)

# Limpieza de filas
df = df[df['price'] != 0]
df = df[df['minimum_nights'] <= 365]

# Relleno de nulos
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Encoding de room_type
room_mapping = {'Shared room': 0, 'Private room': 1, 'Entire home/apt': 2}
df['room_type'] = df['room_type'].map(room_mapping)

# One Hot Encoding de neighbourhood_group
df = pd.get_dummies(df, columns=['neighbourhood_group'])

# Eliminación de outliers
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[df['price'] <= Q3 + 1.5 * IQR]

Q1 = df['minimum_nights'].quantile(0.25)
Q3 = df['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1
df = df[df['minimum_nights'] <= Q3 + 1.5 * IQR]

df = df[df['calculated_host_listings_count'] <= 10]

# Separación de features y target
X = df.drop('price', axis=1)
y = df['price']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar datos procesados
df.to_csv('../data/processed/AB_NYC_2019_processed.csv', index=False, encoding='utf-8')
          
print(f"Pipeline completado. Dataset final: {X_train.shape[0]} filas de entrenamiento y {X_test.shape[0]} filas de test.")
