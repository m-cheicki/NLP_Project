import time
from spellchecker import SpellChecker
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.metrics import *
import nltk
from langdetect import detect
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings('ignore')


nltk.download('words')


def load_sample(PATH='./datasets/', start_range=1, end_range=5):
    start = time.time()

    splitted_datasets = []

    for sample in range(start_range, end_range+1):
        start_load = time.time()

        dataset = pd.read_csv(
            PATH + 'openfoodfacts_part' + str(sample)+'.csv',
            sep='\t')

        end_load = time.time()

        print(f'Sample {sample} : {end_load - start_load} sec.')

        splitted_datasets.append(dataset)

    end = time.time()

    print('-'*20)
    print(f'Load {end_range - start_range + 1} samples : {end - start} sec.')

    return pd.concat(splitted_datasets)


def delete_empty_columns(dataset, rate=0.8):
    columns_to_drop = ['Unnamed: 0', 'url', 'code', 'creator', 'created_t', 'created_datetime', 'last_modified_t',
                       'last_modified_datetime', 'abbreviated_product_name', 'generic_name', 'packaging',
                       'packaging_tags', 'packaging_text', 'brands', 'brands_tags', 'brand_owner', 'categories', 'categories_en', 'origins',
                       'origins_en', 'manufacturing_places', 'labels', 'labels_en', 'emb_codes', 'emb_codes_tags', 'countries', 'countries_tags', 'countries_en',
                       'first_packaging_code_geo', 'cities', 'purchase_places', 'stores', 'countries', 'countries_en',
                       'traces', 'traces_en', 'allergens_en', 'serving_size', 'serving_quantity', 'additives',
                       'additives_en', 'ingredients_from_palm_oil', 'ingredients_that_may_be_from_palm_oil', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n',
                       'states', 'states_tags', 'states_en', 'main_category_en', 'image_small_url', 'image_url',
                       'image_ingredients_url', 'image_ingredients_small_url', 'image_nutrition_url',
                       'image_nutrition_small_url', 'nutrition-score-fr_100g']

    for col in dataset.columns:
        if dataset[col].isna().sum() / len(dataset) > rate:
            columns_to_drop.append(col)

    return delete_specific_columns(dataset, columns_to_drop=columns_to_drop)


def delete_specific_columns(dataset, columns_to_drop=[]):
    columns_to_keep = ['product_name', 'categories_tags', 'ingredients_text', 'additives_tags',
                       'nutriscore_score', 'nutriscore_grade', 'nova_group', 'pnns_groups_1',
                       'pnns_groups_2', 'main_category', 'energy-kcal_100g', 'energy_100g',
                       'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                       'salt_100g']
    cols = []
    for column in columns_to_drop:
        if column in dataset.columns and column not in columns_to_keep:
            cols.append(column)

    print(f'Delete {len(cols)} columns.')

    return dataset.drop(columns=cols)


def correct_enconding_characters(x):
    x = x.replace('\_', '')
    x = x.replace('\%', '')
    x = x.replace('\*', '')

    x = clean_ingredients_list(x)

    x = x.lower()
    x = x.strip()

    x = x.replace('ã©', 'é')
    x = x.replace('&quot;', '')
    x = x.replace('cã¨', 'è')
    x = x.replace('à¨', 'ê')
    x = x.replace('ã', 'à')
    x = x.replace('ã´', 'ô')
    x = x.replace('à´', 'ô')
    x = x.replace('à¢', 'â')
    x = x.replace('à¯', 'ï')
    x = x.replace('à®', 'î')
    x = x.replace('å', 'oe')
    x = x.replace('â', '\'')

    return x


def clean_ingredients_list(x):
    # Delete additives as there is already an 'additive' column
    # Delete vitamins as we are not going to use them
    x = re.sub('(b|e){1}\d*\w', '', x)

    # Delete quantities
    x = re.sub('(\d)+([a-zA-Z])+', '', x)

    return x


def detect_language(x):
    try:
        return detect(x)
    except:
        return "unknown"


def translate(x):
    try:
        return translator.translate(x)
    except:
        return "Cannot translate"


tokenizer = RegexpTokenizer("[A-Za-z'%-]+")
translator = GoogleTranslator(source='auto', target='en')

# Load and clean
dataset = load_sample(start_range=1, end_range=40)
df = delete_empty_columns(dataset)
df = df.dropna(subset=['ingredients_text'])
df['ingredients_text'] = df['ingredients_text'].apply(
    correct_enconding_characters)

# Detect Language
start = time.time()
df['language'] = df["ingredients_text"].apply(detect_language)
end = time.time()

print(f'Detect language : {end - start} seconds...')
print('Detected languages : ' + df['language'].unique())

# Translate
# start = time.time()
# for i, lang in enumerate(df['language']):
#     if lang == 'en':
#         df.at[i, 'ingredients_en'] = df['ingredients_text'].iloc[i]
#     else :
#         df.at[i, 'ingredients_en'] = translate(df['ingredients_text'].iloc[i])
# end = time.time()

# print(f'Translate ingredients : {end - start} seconds...')

# Keep only those in english
df_ingredients_en = df[df['language'] == 'en']
print(f"Loss : {(1 - (df_ingredients_en.shape[0] / df.shape[0])) * 100} %")

# Tokenize
df_ingredients_en['ingredients_text_tokens'] = df_ingredients_en['ingredients_text'].apply(
    lambda x: tokenizer.tokenize(x))


df_ingredients_en.to_csv('./datasets/v1/openfoodfacts.csv', sep='\t')
