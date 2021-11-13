# Import libraries

from collections import Counter
from deep_translator import GoogleTranslator
from langdetect import detect
import nltk
from nltk.metrics import *
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import re
from spellchecker import SpellChecker
import time
import warnings

warnings.filterwarnings('ignore')
nltk.download('words')


def load_sample(start_range=1, end_range=5, PATH='./datasets/'):
    start = time.time()

    splitted_datasets = []

    for sample in range(start_range, end_range+1):
        start_load = time.time()

        dataset = pd.read_csv(
            PATH + 'openfoodfacts_part' + str(sample)+'.csv',
            sep='\t', engine='python')

        end_load = time.time()

        print(f'Sample {sample} : {end_load - start_load} sec.', flush=True)

        splitted_datasets.append(dataset)

    end = time.time()

    print('-'*20)
    print(f'Load {end_range - start_range + 1} samples : {end - start} sec.')

    df = pd.concat(splitted_datasets)
    print(f'Dataset before removing duplicates : {df.shape[0]} entries')

    # df = df.drop_duplicates()
    # print(f'Dataset after removing duplicates : {df.shape[0]} entries')
    return df


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
                       'image_nutrition_small_url']

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
    # Delete percentages
    x = re.sub('\d+\%', '', x)

    return x


def clean_nutrition_facts_for_100g(x):
    if pd.isna(x):
        x = 0
    elif x > 100:
        x = 100
    elif x < 0:
        x = 0
    else:
        x = x
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


def spell_check_nltk(x):
    list_distance = list()
    if x not in english_vocab:
        for _ in english_vocab:
            list_distance.append(edit_distance(_, x))
    return list(english_vocab)[list_distance.index(min(list_distance))]


def spell_check_levenshtein(x):
    misspelled = spell.unknown([x])
    if len(misspelled):
        x = spell.correction(list(misspelled)[0])
    return x


tokenizer = RegexpTokenizer("[A-Za-z\'\%\-]+")
translator = GoogleTranslator(source='auto', target='en')
spell = SpellChecker()
ps = PorterStemmer()

############
# Load and clean
############

dataset = load_sample(start_range=1, end_range=40)
df = delete_empty_columns(dataset)

# Clean ingredients
df = df.dropna(subset=['ingredients_text'])
df['ingredients_text'] = df['ingredients_text'].apply(
    correct_enconding_characters)

# Cleaning nutrition facts
df['fat_100g'] = df['fat_100g'].apply(clean_nutrition_facts_for_100g)
df['saturated-fat_100g'] = df['saturated-fat_100g'].apply(
    clean_nutrition_facts_for_100g)
df['carbohydrates_100g'] = df['carbohydrates_100g'].apply(
    clean_nutrition_facts_for_100g)
df['sugars_100g'] = df['sugars_100g'].apply(clean_nutrition_facts_for_100g)
df['fiber_100g'] = df['fiber_100g'].apply(clean_nutrition_facts_for_100g)
df['proteins_100g'] = df['proteins_100g'].apply(clean_nutrition_facts_for_100g)
df['salt_100g'] = df['salt_100g'].apply(clean_nutrition_facts_for_100g)

############
# Detect Language
############

start = time.time()
df['language'] = df["ingredients_text"].apply(detect_language)
end = time.time()

print(f'Detect language : {end - start} seconds...')
print('Detected languages : ' + df['language'].unique())

############
# Translate
############

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


############
# Tokenize
############

df_ingredients_en["ingredients_token"] = df_ingredients_en["ingredients_text"].apply(
    lambda x: tokenizer.tokenize(x))


df_ingredients_en.to_csv('./datasets/v2/openfoodfacts.csv', sep='\t')


############
# Spelling check
############
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
ingredient_list = [
    _ for list in df_ingredients_en["ingredients_token"].to_list() for _ in list]
ingredient_list.sort()
set_ingredients = set(ingredient_list)

df_spelling_ingredients = pd.DataFrame(set_ingredients, columns=['Initial'])
df_spelling_ingredients

# NLTK
corrects = []
start_time = time.time()
for word in list(set_ingredients):
    list_distance = list()
    if word not in english_vocab:
        for _ in english_vocab:
            list_distance.append(edit_distance(_, word))
        correct = list(english_vocab)[list_distance.index(min(list_distance))]
        corrects.append(correct)
        print(f"{word} ==> {correct}", flush=True)

end_time = time.time()

df_spelling_ingredients['NLTK'] = corrects

print(f"Spelling mistakes - Method 1 : {end_time - start_time} seconds.")

# SpellChecker using Levenshtein
corrects = []
start_time = time.time()
for _ in set_ingredients:
    misspelled = spell.unknown([_])
    if len(misspelled):
        correct = spell.correction(list(misspelled)[0])
        print(f"{_} ==> {correct}")
        corrects.append(correct)

df_spelling_ingredients['Levenshtein'] = corrects

end_time = time.time()
print(f"Spelling mistakes - Method 2 : {end_time - start_time} seconds.")

# Word frequency : manual
occ = Counter(ingredient_list)

print("Word frequency : ")
print(occ)
print(len(occ))

# Word frequency with stemming
ingredients_stemmed = []
for ingredient in ingredient_list:
    try:
        ingredients_stemmed.append(ps.stem(ingredient))
    except:
        ingredients_stemmed.append("*"+ingredient+"*")
occ_stemmed = Counter(ingredients_stemmed)

print("Word frequency using stemming: ")
print(occ_stemmed)
print(len(occ_stemmed))
