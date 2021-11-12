# Advanced Machine Learning for NLP and Text Processing

## Project 1 : OpenFoodFacts

OpenFoodFacts can be considerate of a wikipedia for food !
The goal of OpenFoodFacts is to share with everyone a maximum of informations on food products.
It contains more than 800000 products but maybe all products are not perfectly described...
Mainly, for a product, we can find the list of ingredients, nutrition facts and food categories.

### Define and clean the vocabulary of ingredients 

- Did you find some mistakes ? 
- How did you manage them ? 

Our code implementation for this question are in the <a href="./CELIE_CHEICKISMAIL_OpenFoodFacts_part1.ipynb">CELIE_CHEICKISMAIL_OpenFoodFacts_part1.ipynb</a> notebook in this repository. 

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

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries. 

Observations :

This method is quite simple to use. We retrieve words from the NLTK's corpora and we compare them to the words we have. But by doing so, we have got some unexpected behaviors : 
- even though we put a condition saying that if the element we want to check is already in the vocabulary, you don't have to check the spelling (if the word exists, this word is already correctly spelled), sometimes it checks : `green ==> green`. Hopefully, it gives the same results which is nice. 
- it doesn't recognise some words like `vinegar`. In our tests, we had `vingar ==> dingar` which means _The giant honeybee, Apis dorsata, native to South and South-East Asia, which is noted for aggressive defence of its nests, which typically hang from trees, cliffs, and buildings._ It can be understandable as there is only one letter difference `v ==> d`

There are also other issues but these are because of the dataset, for instance : 
- `containslessthanof` should be `contains less than of` 
- Some words are correctly spelled but the correction is not pertinent : `all-purpose ==> apurpose`, `hours ==> cours`
- Even though we were very specific by filtering by language, keeping only those in English, some French words appeared and the vocabulary couldn't check the spelling correctly : `farine ==> parine`


Although, there are issues, some words were correctly spelled like `monfat ==> nonfat`, `northn ==> north`... 

```
vanillan ==> vanilla
monfat ==> nonfat
lychs ==> lycus
trate ==> irate
carrangnan ==> caranna
lutr ==> lut
varying ==> warding
exct ==> exact
diglcid ==> diacid
farine ==> parine
un-ch ==> nunch
hucklry ==> hackery
aglycid ==> glycid
ard ==> dard
salisry ==> salish
dipotass ==> potass
knal ==> knap
goats' ==> goaty
butyloctyl ==> tylostyle
ocesses ==> acestes
hydroylz ==> hydrol
aflavors ==> flavory
dipot ==> divot
mst ==> tst
northn ==> north
containslessthanof ==> curtainless
vingar ==> dingar
aminos ==> minos
...
starcn ==> starch
isugar ==> sugar
cocoapowd ==> cocowood
ithout ==> without
phophate ==> phosphate
```

We couldn't run this method longer, it takes too much time to process. And the results were not as good as wanted.

##### SpellChecker

Pure Python Spell Checking based on Peter Norvig’s blog post on setting up a simple spell checking algorithm.

It uses a Levenshtein Distance algorithm to find permutations within an edit distance of 2 from the original word. It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a word frequency list. Those words that are found more often in the frequency list are more likely the correct results.

Observations :

This method is as easy to use as the first one. 
First, we check if the word is misspelled. If it is, we use the correction function that proposes multiple choices ordered by descending probabilities. For our tests, we have taken the first choice as it is the highest probability to be the correct spelling we want. 

As expected, we found some issues/incoherence due to the dataset: 
- `exct ==> exact` instead of `extract`  
- `farine ==> marine` from a french word

And sometimes, it says that the initial word is misspelled although it is not : `dipotass ==> dipotass`, `carrangnan ==> carrangnan`. It is not problematic but it takes some additional process time. 

But globally, it does pretty well its job : 
```
vanillan ==> vanilla
monfat ==> nonfat
lychs ==> lochs
trate ==> trade
carrangnan ==> carrangnan
lutr ==> lute
exct ==> exact
diglcid ==> diploid
farine ==> marine
un-ch ==> bunch
hucklry ==> hickory
aglycid ==> aglycid
salisry ==> satisfy
dipotass ==> dipotass
knal ==> anal
goats' ==> goats
butyloctyl ==> butyloctyl
ocesses ==> dresses
hydroylz ==> hydroxyl
aflavors ==> flavors
dipot ==> depot
mst ==> must
northn ==> north
containslessthanof ==> containslessthanof
vingar ==> vinegar
aminos ==> amino
lgian ==> lian
spaning ==> sparing
comtry ==> country
swai ==> swap
frk ==> fry
uals ==> pals
stl-cut ==> stl-cut
zn ==> in
onctrate ==> nitrate
granulat ==> granular
arul ==> aru
acontains ==> contains
ngre ==> ogre
amountisving ==> amountisving
jarlsrg ==> jarlsrg
isotyrate ==> isotyrate
lithin-anulsifi ==> lithin-anulsifi
sorae ==> sore
cocao ==> cocoa
cholula ==> cholla
lactlate ==> lactate
rirose ==> hirose
xanthai ==> bantha
sodiumphosphat ==> sodiumphosphat
alumina ==> alumna
anch ==> inch
monocalcium ==> monocalcium
convtiona ==> convtiona
currart ==> currant
hci ==> hi
cormn ==> corn
gattigte ==> gattigte
sike ==> like
cantre ==> centre
ouality ==> quality
powd ==> pod
lilhin ==> within
thc-arm ==> thc-arm
goonlijk ==> goonlijk
ylowfin ==> yellowfin
suf ==> sun
hagisutiyfat ==> hagisutiyfat
xanthang ==> anthing
canfeine ==> caffeine
uil ==> oil
chflour ==> colour
corn-ack ==> corn-ack
cyclohasiloxane ==> cyclohasiloxane
spr ==> sir
sning ==> sing
agt ==> at
...
```

About the process time, this method is way faster than the first one. Also more accurate, we can say that this method better than the first one. 

##### Manual approach

For this approach, we have decided to correct misspelled word using word frequency. The more the word is present the more it has to be the correct spelling. 

For instance : 

```
'alchol': 1,
...
'alcohol': 448,
'alcoholic': 6,
```
We can correct `alchol` and `alcoholic` by `alcohol`. 

Another example is : 

```
'annaito': 1,
'annat': 2,
'annato': 20,
'annatt': 1,
'annatto': 2626,
```

We can correct all these words by `annatto`.

Observations :

This method is quite simple to put in place : you put all word in a lit that you sort by alphabetical order and then you count all occurences. But it has its limits. To easily compare and have better results, misspelled letter should be in the end of the word. For example, if we want to correct `acontains` by `contains` we can't as `a` and `c` are not the same letter. The task becomes too fastidious as the dataset is very big. 

##### To go further 

There are some steps that we haven't tried but we could have : 

- Lemmatization : consists in finding the root of inflected verbs and reducing plural and/or feminine words to the masculine singular form.
- Stemming : consists in keeping only the root of the word


We have decided to test stemming. By doing so, we can regroup words by their roots and then correct their spelling. For example, `alcohol` and `alcoholic` can be grouped in `alcohol`

_Reminder:_ Our code implementation for this question are in the <a href="./CELIE_CHEICKISMAIL_OpenFoodFacts_part1.ipynb">CELIE_CHEICKISMAIL_OpenFoodFacts_part1.ipynb</a> notebook in this repository. 

