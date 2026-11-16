# Copyright 2025 Tsinghua SPMI Lab
# Author: Author: Sardar (sar_dar@foxmail.com)

# This script produces a lexicon via Pinyin to IPA method.
# FROM:
#     word_list: path of word list file
#	  generate Pinyin use pypinyin toolkit, refer to https://pypi.org/project/pypinyin
#     generate IPA use pinyin-to-ipa toolkit, refer to https://pypi.org/project/pinyin-to-ipa
# TO: 
#     lexicon.txt

import os, sys, argparse

try:
	import pypinyin
except ModuleNotFoundError as e:
	print(f"ModuleNotFoundError: need to install pypinyin. refer to https://pypi.org/project/pypinyin")
	sys.exit(1)
try:
	from pinyin_to_ipa import pinyin_to_ipa
except ModuleNotFoundError as e:
	print(f"ModuleNotFoundError: need to install pinyin-to-ipa. refer to https://pypi.org/project/pinyin-to-ipa")
	sys.exit(1)

def ipa_normalize(ipa_seq: str, replace_natual_tone:bool=False) -> str:
	"""
	reference:
		- [wikipedia](https://en.wikipedia.org/wiki/Pinyin)
		- [普通话 IPA 转换器](https://www.notch1p.xyz/putonghua/#:~:text=普通话%20IPA,转换器，为中文文本标注普通话发音的国际音标，可选%2011%20种版本%E3%80%82)
	"""
	replacements = {
		"ai˧˩˧": "a˧˩˧ i", "ai̯˧˩˧": "a˧˩˧ i", "ai̯˧˥": "a˧˥ i", "ai̯˥˩": "a˥˩ i", "ai̯˥": "a˥ i", "ai̯": "a i",
		"au̯˧˩˧": "a˧˩˧ u", "au̯˧˥": "a˧˥ u", "au̯˥˩": "a˥˩ u", "au̯˥": "a˥ u", "au̯": "a u", 
		"ei̯˧˩˧": "e˧˩˧ i", "ei̯˧˥": "e˧˥ i", "ei̯˥˩": "e˥˩ i", "ei̯˥": "e˥ i", "ei̯": "e i", 
		"ou̯˧˩˧": "o˧˩˧ u", "ou̯˧˥": "o˧˥ u", "ou̯˥˩": "o˥˩ u", "ou̯˥": "o˥ u", "ou̯": "o u",
		"ɚ˧˩˧": "ə˧˩˧ ɻ", "ɚ˧˥": "ə˧˥ ɻ", "ɚ˥˩": "ə˥˩ ɻ", "ɚ˥": "ə˥ ɻ", "ɚ": "ə ɻ", 
		"ɔ": "o"
	}

	natual_tone_phones = ["a", "e", "i", "o", "u", "y", "ə", "ɛ", "ɤ", "ɹ̩", "ɻ̩", "ʊ"]

	for old, new in replacements.items():
		ipa_seq = ipa_seq.replace(old, new)
	
	if replace_natual_tone:
		new_ipa_seq = []
		for phone in ipa_seq.split():
			new_phone = ''
			if phone in natual_tone_phones:
			# Replace natural tones with first tone
				new_phone = phone + "˥"
			else:
				new_phone = phone
			new_ipa_seq.append(new_phone)
		return ' '.join(new_ipa_seq)
	else:
		return ipa_seq


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script produces a lexicon via Pinyin to IPA method.")
	parser.add_argument("word_list_file", type=str, help="path of word list file")
	parser.add_argument("--replace_natual_tone", "-r", action="store_true", help="Whether replace natural tone to first tone. default: False")
	parser.add_argument("--out", type=str, default=None, help="path of output lexicon file. default: lexicon.txt")
	args = parser.parse_args()
	
	assert os.path.isfile(args.word_list_file), f"{args.word_list_file} is not exist!"
	out = args.out if args.out else os.path.join(os.path.dirname(args.word_list_file), "lexicon.txt")

	with open(out, 'w', encoding='utf-8') as wf, open(args.word_list_file, 'r', encoding='utf-8') as f:
		for line in f:
			word = line.strip()
			# For polyphonic characters (多音字), only the first pronunciation is taken.
			pinyin_seq = pypinyin.pinyin(word, style=pypinyin.Style.TONE3) 
			ipa_list = []
			for pinyin in pinyin_seq:
				try:
					pron_ipa = pinyin_to_ipa(pinyin[0])
				except ValueError as e:
					print(f"{type(e).__name__}: occured a invalid pinyin: {pinyin[0]}, when converting word: {word}")
					continue
					# sys.exit(1)
				ipa_list.extend(list(pron_ipa[0]))
			
			ipa_norm = ipa_normalize(' '.join(ipa_list), replace_natual_tone=args.replace_natual_tone)
			wf.write(word + '\t' + ipa_norm + '\n')
