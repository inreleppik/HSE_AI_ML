
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin

class CustomCarDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        
        self.medians = {
            'mileage': 19.4,
            'engine': 1248.0,
            'max_power': 81.86,
            'seats': 5,
            'torque': 160,
            'max_torque_rpm': 3000
        }
        self.train_cols = [
            'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque','max_torque_rpm', 
            'fuel_CNG','fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 
            'seller_type_Dealer', 'seller_type_Individual','seller_type_Trustmark Dealer', 
            'transmission_Automatic', 'transmission_Manual',
            'owner_First Owner', 'owner_Fourth & Above Owner', 'owner_Second Owner',
            'owner_Test Drive Car', 'owner_Third Owner', 
            'car_brand_Ambassador', 'car_brand_Audi', 'car_brand_BMW', 'car_brand_Chevrolet', 'car_brand_Daewoo',
            'car_brand_Datsun', 'car_brand_Fiat', 'car_brand_Force', 'car_brand_Ford', 'car_brand_Honda', 
            'car_brand_Hyundai', 'car_brand_Isuzu', 'car_brand_Jaguar', 'car_brand_Jeep', 'car_brand_Kia', 
            'car_brand_Land','car_brand_Lexus', 'car_brand_MG', 'car_brand_Mahindra','car_brand_Maruti',
            'car_brand_Mercedes-Benz', 'car_brand_Mitsubishi','car_brand_Nissan', 'car_brand_Peugeot', 
            'car_brand_Renault', 'car_brand_Skoda','car_brand_Tata', 'car_brand_Toyota', 'car_brand_Volkswagen',
            'car_brand_Volvo', 'seats_2','seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9', 'seats_10', 'seats_14'
        ]

        self.drop_cols = [
            'fuel_Diesel', 'seller_type_Individual', 'transmission_Manual', 
            'owner_First Owner', 'car_brand_Maruti', 'seats_5'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        data = X.copy()

        data['mileage'] = (
            data.mileage.fillna('0')
            .apply(lambda x: x.strip(' kmpl').strip(' km/kg'))
            .astype(float)
            .replace(0, np.nan)
        )
        data['engine'] = (
            data.engine.fillna('0')
            .apply(lambda x: x.strip(' CC'))
            .astype(float)
            .replace(0, np.nan)
        )
        data['max_power'] = (
            data.max_power.fillna('0')
            .apply(lambda x: x.strip(' bhp'))
            .replace('', '0')
            .astype(float)
            .replace(0, np.nan)
        )

        torque_vals = []
        max_torque_rpm_vals = []
        series = data.torque.fillna('0').apply(lambda x: x.replace(',', '')).apply(str.lower)

        for i in range(len(data)):
            numbers = re.findall(r'\d+\.?\d*', series.iloc[i])
            if 'kgm' in series.iloc[i]:
                # 1 кгм ≈ 9.8 Н·м
                torque_vals.append(float(numbers[0]) * 9.8 if numbers else np.nan)
            elif 'nm' in series.iloc[i]:
                torque_vals.append(float(numbers[0]) if numbers else np.nan)
            else:
                torque_vals.append(np.nan)

            if len(numbers) >= 2:
                max_torque_rpm_vals.append(float(numbers[-1]))
            else:
                max_torque_rpm_vals.append(np.nan)

        data['torque'] = torque_vals
        data['max_torque_rpm'] = max_torque_rpm_vals
        # 
        data.loc[data.torque > 800, 'torque'] = data.loc[data.torque > 800, 'torque'] / 9.8
        data.loc[data.torque > 700, 'torque'] = data.loc[data.torque > 700, 'torque'] / 10
        data.loc[data.max_torque_rpm > 10000, 'max_torque_rpm'] = data.loc[data.max_torque_rpm > 10000, 'max_torque_rpm'] / 10

        # Заполнение пропусков с помощью медиан
        na_cols = ['mileage', 'engine', 'max_power', 'seats', 'torque', 'max_torque_rpm']
        data[na_cols] = data[na_cols].fillna(self.medians)

        # Приведение типов
        data = data.astype({'engine': int, 'seats': int}, errors='ignore')

        # Добавляем бренды автомобилей и удаляем колонку name
        data['car_brand'] = data.name.str.strip().str.split().str[0]
        data.drop(columns=['name'], inplace=True, errors='ignore')

        # Логарифмируем некоторые столбцы для приведения зависимости к более линейному виду
        data['engine'] = data['engine'].apply(np.log)
        data['torque'] = data['torque'].apply(np.log)
        data['max_power'] = data['max_power'].apply(np.log)

        # OHE-кодирование
        data_cat = pd.get_dummies(
            data, 
            columns=['fuel', 'seller_type', 'transmission', 'owner', 'car_brand', 'seats'],
            drop_first=False  # Мы сами выберем, какие дропать
        )

        # Добавление колонок из тренировочных данных и удаление колонок, которые не использовались при обучении модели
        missing_cols = set(self.train_cols) - set(data_cat.columns)
        for col in missing_cols:
            data_cat[col] = 0
        
        extra_cols = set(data_cat.columns) - set(self.train_cols)
        data_cat.drop(columns=list(extra_cols), inplace=True)

        # Удаляем столбцы базовых категорий
        data_cat.drop(columns=self.drop_cols, inplace=True, errors='ignore')

        # Добавим правильный порядок столбцов
        data_cat = data_cat[self.get_feature_names_out()]

        return data_cat
    
    def get_feature_names_out(self):
        # Убираем из train_cols те, что дропаются, и оставляем правильный порядок
        final_cols = [col for col in self.train_cols if col not in self.drop_cols]
        return final_cols
