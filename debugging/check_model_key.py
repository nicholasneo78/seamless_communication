"""
code to debug the missing keys in the state model dict
"""

import json

KEY_MISSING_DIR = '/seamless_communication/debugging/key_missing.json'
KEY_UNEXPECTED_DIR = '/seamless_communication/debugging/key_unexpected.json'

# load json files
with open(KEY_MISSING_DIR, 'r+') as f_missing:
    key_missing_list = json.load(f_missing)

with open(KEY_UNEXPECTED_DIR, 'r+') as f_unexpected:
    key_unexpected_list = json.load(f_unexpected)

print(len(key_missing_list))
print(len(key_unexpected_list))

# do a matchup between similar model keys to find out what are the extra keys
key_missing_list = ['model.'+key for key in key_missing_list]
# print(key_missing_list)

difference = (set(key_missing_list) | set(key_unexpected_list)) - (set(key_missing_list) & set(key_unexpected_list))

print(difference)
print(len(list(difference)))