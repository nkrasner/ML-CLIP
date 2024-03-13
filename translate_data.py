# import googletrans
# import json
from tqdm import tqdm

# translator = googletrans.Translator()

# langs = ["ja", "es", "hi"]

# with open("data/annotations/captions_train2017_ordered.json", "r") as f:
#     captions = json.load(f)
    
# for lang in langs:
#     print("Translating to {}...".format(lang))
#     trans_captions = []
#     for caption in tqdm(captions):
#         translated = translator.translate(caption, dest=lang)
#         trans_captions.append(translated.text)
#     with open("data/captions_{}.json".format(lang), "w") as f:
#         json.dump(trans_captions, f)


import json
import googletrans
import time

# #################################################################################################################
# Used to generate the ordered jsons
# #################################################################################################################


# json_file_path = 'data/annotations/captions_val2017.json'
#
# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)
#
#
# id_annotations = {k['image_id']: k for k in data['annotations']}
#
# ordered_list = list(id_annotations.keys())
# ordered_list.sort()
#
# ordered_annotations = {k: id_annotations[k] for k in ordered_list}
#
# json_file_path = 'data/annotations/captions_val2017_ordered.json'
#
# # Open the file in write mode and write the dictionary to it
# with open(json_file_path, 'w') as json_file:
#     json.dump(ordered_annotations, json_file, indent=2)


# #################################################################################################################
# Translating testing
# #################################################################################################################

json_file_path = 'data/annotations/captions_val2017_ordered_translated.json'

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# print(data[list(data.keys())[0]])
trans = googletrans.Translator()

new_data = []
# count = 0
# total = len(data.keys())
for example in tqdm(data):
# for k in tqdm(data.keys()):
    # cap_dict = data[k]
    cap_dict = example
    # cap = cap_dict.pop('caption')
    cap = cap_dict['captions']['en']
    # cap_dict["captions"] = {}
    # cap_dict["captions"]['en'] = cap
    
    while True:
        try:
            # es_text = trans.translate(cap, dest='es').text
            # ja_text = trans.translate(cap, dest='ja').text
            # hi_text = trans.translate(cap, dest='hi').text
            fr_text = trans.translate(cap, dest='fr').text
            de_text = trans.translate(cap, dest='de').text
            zh_text = trans.translate(cap, dest='zh-cn').text
            bn_text = trans.translate(cap, dest='bn').text
            break
        except:
            time.sleep(5)
            continue
    

    # cap_dict["captions"]['es'] = es_text
    # cap_dict["captions"]['ja'] = ja_text
    # cap_dict["captions"]['hi'] = hi_text
    cap_dict["captions"]['fr'] = fr_text
    cap_dict["captions"]['de'] = de_text
    cap_dict["captions"]['zh'] = zh_text
    cap_dict["captions"]['bn'] = bn_text

    new_data.append(cap_dict)
    # count = count + 1
    # print(f'finished: {count/total}')

# print(data[list(data.keys())[0]])

json_file_path = 'data/annotations/captions_val2017_ordered_translated_with_unseen.json'

# Open the file in write mode and write the dictionary to it
with open(json_file_path, 'w') as json_file:
    json.dump(new_data, json_file, indent=2)