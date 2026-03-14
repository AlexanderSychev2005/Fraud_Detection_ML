import pandas as pd
import numpy as np
import time

# Если у вас не установлен tqdm, установите: pip install tqdm
from tqdm import tqdm
tqdm.pandas() # Включает поддержку прогресс-баров для pandas .apply()

print("🚀 Запуск генерации признаков (Feature Engineering)...\n")
start_time_total = time.time()

# ==============================================================================
# 0. БАЗОВАЯ ПОДГОТОВКА И СОРТИРОВКА (КРИТИЧЕСКИ ВАЖНО)
# ==============================================================================
print("[1/7] ⏳ Загрузка датасета 'merged_full_dataset.csv'...")
t0 = time.time()
df = pd.read_csv("merged_full_dataset.csv")
print(f"      ✅ Загружено строк: {len(df)}. Заняло: {time.time() - t0:.2f} сек.")

print("[2/7] ⏳ Конвертация дат и сортировка...")
t0 = time.time()
df['timestamp_tr'] = pd.to_datetime(df['timestamp_tr'], errors='coerce')
df['timestamp_reg'] = pd.to_datetime(df['timestamp_reg'], errors='coerce')

df = df.sort_values(by=['id_user', 'timestamp_tr']).reset_index(drop=True)

string_columns = ['card_holder', 'email', 'card_type', 'status', 'error_group']
# Проверяем, какие колонки реально есть в df, чтобы не было ошибки
string_columns_exist = [col for col in string_columns if col in df.columns]
df[string_columns_exist] = df[string_columns_exist].fillna('UNKNOWN')
print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")

# ==============================================================================
# 1. ГЕОГРАФИЧЕСКИЕ НЕСОВПАДЕНИЯ (Geo-Mismatch)
# ==============================================================================
print("[3/7] 🌍 Генерация гео-признаков...")
t0 = time.time()

df['is_reg_pay_mismatch'] = (df['reg_country'] != df['payment_country']).astype(int)
df['is_card_pay_mismatch'] = (df['card_country'] != df['payment_country']).astype(int)
df['is_reg_card_mismatch'] = (df['reg_country'] != df['card_country']).astype(int)
df['geo_mismatch_score'] = df['is_reg_pay_mismatch'] + df['is_card_pay_mismatch'] + df['is_reg_card_mismatch']

print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")

# ==============================================================================
# 2. АНАЛИЗ ИМЕНИ И ПОЧТЫ (Identity & Email)
# ==============================================================================
print("[4/7] 🕵️‍♂️ Анализ имени и почты (самый долгий этап)...")
t0 = time.time()

df['email_domain'] = df['email'].astype(str).str.split('@').str[1]
df['email_digits_count'] = df['email'].astype(str).str.count(r'\d')

def check_name_in_email(row):
    if pd.isna(row['card_holder']) or pd.isna(row['email']) or row['card_holder'] == 'UNKNOWN' or row['email'] == 'UNKNOWN':
        return 0
    first_name = str(row['card_holder']).split()[0].lower()
    return 1 if first_name in str(row['email']).lower() else 0

# Используем progress_apply вместо обычного apply, чтобы видеть ползунок загрузки!
print("      Запуск проверки совпадений (progress_apply):")
df['name_in_email'] = df.progress_apply(check_name_in_email, axis=1)

suspicious_names = ['UNKNOWN', 'VALUED CUSTOMER', 'CARDHOLDER', 'NO NAME']
df['is_suspicious_cardholder'] = df['card_holder'].astype(str).str.upper().isin(suspicious_names).astype(int)

print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")

# ==============================================================================
# 3. КАРТОЧНЫЕ АНОМАЛИИ И АГРЕГАЦИИ (Card Anomalies)
# ==============================================================================
print("[5/7] 💳 Расчет карточных аномалий...")
t0 = time.time()

df['users_per_card'] = df.groupby('card_mask_hash')['id_user'].transform('nunique')
df['is_prepaid_card'] = df['card_type'].astype(str).str.lower().isin(['prepaid', 'virtual']).astype(int)

print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")

# ==============================================================================
# 4. ВРЕМЕННЫЕ И СКОРОСТНЫЕ ПРИЗНАКИ (Time & Velocity)
# ==============================================================================
print("[6/7] ⏱ Генерация временных и скоростных признаков...")
t0 = time.time()

df['mins_since_reg'] = (df['timestamp_tr'] - df['timestamp_reg']).dt.total_seconds() / 60.0

df['seconds_since_last_trans'] = df.groupby('id_user')['timestamp_tr'].diff().dt.total_seconds()
df['seconds_since_last_trans'] = df['seconds_since_last_trans'].fillna(-1)

df['trans_hour'] = df['timestamp_tr'].dt.hour
df['trans_day_of_week'] = df['timestamp_tr'].dt.dayofweek

print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")

# ==============================================================================
# 5. ДЕНЕЖНЫЕ И ОШИБОЧНЫЕ ПАТТЕРНЫ (Amount & Errors)
# ==============================================================================
print("[7/7] 💰 Расчет денежных и поведенческих паттернов...")
t0 = time.time()

df['is_micro_payment'] = (df['amount'] < 2.0).astype(int)
df['is_round_amount'] = (df['amount'] % 1 == 0).astype(int)

user_avg_amount = df.groupby('id_user')['amount'].transform('mean')
df['amount_vs_user_avg'] = df['amount'] / (user_avg_amount + 0.001)

df['user_trans_sequence'] = df.groupby('id_user').cumcount() + 1

df['is_fail'] = (~df['status'].astype(str).str.lower().isin(['success', 'approved'])).astype(int)
df['cumulative_fails_before'] = df.groupby('id_user')['is_fail'].cumsum() - df['is_fail']

df['is_cvv_error'] = df['error_group'].astype(str).str.lower().str.contains('cvv', na=False).astype(int)
df['has_cvv_error_history'] = df.groupby('id_user')['is_cvv_error'].cummax() - df['is_cvv_error']

df.drop(columns=['is_fail', 'is_cvv_error'], inplace=True, errors='ignore')

print(f"      ✅ Выполнено за {time.time() - t0:.2f} сек.")
# Сохраняем датафрейм со всеми новыми фичами в НОВЫЙ файл
df.to_csv("dataset_with_features.csv", index=False)

print("💾 Готово! Данные успешно сохранены в файл 'dataset_with_features.csv'")
# ==============================================================================
# ФИНАЛ
# ==============================================================================
total_time = (time.time() - start_time_total) / 60.0
print("-" * 50)
print(f"🎉 ВСЕ ГОТОВО! Новых признаков сгенерировано.")
print(f"Новый размер датасета: {df.shape}")
print(f"Общее время выполнения: {total_time:.2f} минут.")
print("-" * 50)

# df.to_csv("dataset_with_features.csv", index=False)
# print("Данные сохранены в 'dataset_with_features.csv'")