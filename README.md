# Advanced Machine Learning for NLP and Text Processing

## Project 1 : OpenFoodFacts

OpenFoodFacts can be considerate of a wikipedia for food !
The goal of OpenFoodFacts is to share with everyone a maximum of informations on food products.
It contains more than 800000 products but maybe all products are not perfectly described...
Mainly, for a product, we can find the list of ingredients, nutrition facts and food categories.

### Define and clean the vocabulary of ingredients 

- Did you find some mistakes ? 
- How did you manage them ? 

First, this dataset is composed of almost 2 millions of lines and 187 columns. 
Here is the <a href="https://static.openfoodfacts.org/data/data-fields.txt">documentation</a> of the dataset. <br/>

_**Note**: Note that this documentation is not up-to-date, some columns are not present in the docs but are in the dataset. Also, there is some incoherence we have faced. For instance, in the docs it is said that the CSV containing the dataset is encoded in UTF-8 but when we have imported it we found some encoding issues (we can't say if this is from the data itself before being exported into CSV file, or during the export of the dataset)._

#### Cleaning dataset 

Before answering this question some preprocessing steps have been done. 

Importing 2M lines can be quite fat but having some processing steps on all these lines can be very fastidious. So, we have splitted the initial dataset into 40 samples. 

Also, lots of columns were either empty or almost empty. Thus, for the purpose of this project, some columns/data weren't necesssary. <br/>

So by default, we have decided to delete these columns as we are not going to use them : 
```
['Unnamed: 0', 'url', 'code', 'creator', 'created_t', 'created_datetime', 'last_modified_t','last_modified_datetime', 'abbreviated_product_name', 'generic_name', 'packaging', 'packaging_tags', 'packaging_text', 'brands', 'brands_tags', 'brand_owner', 'categories', 'categories_en', 'origins','origins_en', 'manufacturing_places', 'labels', 'labels_en', 'emb_codes', 'emb_codes_tags', 'countries', 'countries_tags', 'countries_en','first_packaging_code_geo', 'cities', 'purchase_places', 'stores', 'countries', 'countries_en', 'traces', 'traces_en', 'allergens_en', 'serving_size', 'serving_quantity', 'additives', 'additives_en', 'ingredients_from_palm_oil', 'ingredients_that_may_be_from_palm_oil', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n', 'states', 'states_tags', 'states_en', 'main_category_en', 'image_small_url', 'image_url', 'image_ingredients_url', 'image_ingredients_small_url', 'image_nutrition_url', 'image_nutrition_small_url']
```

Then, we droped columns that were filled with less than 20% of the whole dataset.

In the meantime, we also wanted columns to be mandatory : 
```
'product_name', 'categories_tags', 'ingredients_text', 'additives_tags', 'nutriscore_score', 'nutriscore_grade', 'nova_group', 'pnns_groups_1', 'pnns_groups_2', 'main_category', 'energy-kcal_100g', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g'
```

These columns represent : 
- Product name
- Food Categories and Main categories
- List of ingredients
- Food Additves
- Nutriscore (score and grade)
- Nova group
- PNNS groups (large and sub-categories)
- Nutrition Facts (the most common in our packages, at least)

After deleting columns, we wanted to explore deeper our dataset and we have found out that some encoding issues were present. To correct them, we have replaced manually some problematic characters. <br/>
During our tests, we have checked the encoding row per row of the dataset and we found that we had `UTF-8` as mentionned in the docs but also `latin1`, `ascii`, `iso-8859-1`, `cp1252`... which can explain weird and special characters. 

We have also seen that in the ingredient list, some information contains quantities, percentages or additves (and others digit information) which was problematic for the next steps, so we have deleted them at the same time. By doing so, we could say that there is loss of information but additives have been drawn up in the `additives_tags` column, some quantities (grams and percentages) are sometimes mentionned in the corresponding column (+ we will see that we are not going to use them as we have nutrition facts for 100g). 

#### Detect languages

As OpenFoodFacts is a project developed by thousands of volunteers accross the world, we can find products comming from different countries, which also means in different languages. Even though we have taken the France link to have French products only (as we thought at first) we found out that some products are in different languages like english, spanish, portugese, italian... 

In order to know better our dataset and filter afterwards, we wanted to detect automatically the language of each product. To do so, we have used different packages to test them and finally chose `langdetect`. 

In our tests, it took 2h15 to detect the language of almost 2M entries. 

**Note**: It is not perfectly accurate but it is doing pretty well its job. Some issues occured in this step because some ingredients started with digits or were links and you cannot predict a language on numbers and URL (yes, some people found it funny to put some links in the ingredient links from TikTok, which unfortunately does not work :'(, or from commercials). 

For the further steps, we have taken two approaches : 
- Either take only those in english as it is easier to check spelling mistakes 
- Or translate every row that are not in english and make the dataset unilingual

These two approaches have their own pros and cons : 
- First approach : 
    - Main pro : unilingual (English) and its vocabulary/dictionnaries are well developped
    - Main con : this means a big loss of our dataset as only 10% of the dataset is in English
- Second approach : 
    - Main pro : unilingual and without loss of many entries of our dataset
    - Main cons : 
        - Finding APIs/packages that does the trick (some have character limitation)
        - Accuracy of the translation ? 
        - Time of processing (tried in a little sample, by calculus we found that we needed almost 10 days non-stop to translate every row, which we can't afford to do)

In our case, we decided to go for the first option : keep only those in English. 

#### Handling mistakes

For this part, we have taken three different approaches : 
- Use NLTK's corpus vocabulary
- Use SpellChecker, a python package
- A manual approach

##### NLTK corpus vocabulary

##### SpellChecker

##### Manual approach

Our code implementation for this question are in the **CELIE_CHEICKISMAIL_OpenFoodFacts_part1.ipynb** notebook in this repository. 

