import os
import argparse


def load_lexicon(lexicon_file, force_cased=False):
	"""
	从词典文件中加载词汇
	:param lexicon_file: 词典文件路径
	:return: 包含词典中所有单词的集合
	"""
	word_list = set()

	with open(lexicon_file, 'r', encoding='utf-8') as f:
		for line in f:
			word = line.strip().split(maxsplit=1)[0]
			if force_cased:
				word_list.add(word.upper())
			else:
				word_list.add(word)
	return word_list


def find_oov_words(text_file, word_list, force_cased=False):
	"""
	找出文本文件中的 OOV 单词，并计算 OOV 率
	:param text_file: 文本文件路径
	:param word_list: 词表
	:return: OOV 单词集合和 OOV 率
	"""
	oov_words = set()
	all_words = set()
	count_oov_words_with_repeated = 0
	count_all_words_with_repeated = 0
	for text in text_file:
		with open(text, 'r', encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split(maxsplit=1)
				if len(parts) > 1:
					if force_cased:
						sequence = parts[1].upper()
					else:
						sequence = parts[1]
					words = sequence.split()
					all_words.update(words)
					count_all_words_with_repeated += len(words)
					for word in words:
						if word not in word_list:
							oov_words.add(word)
							count_oov_words_with_repeated += 1

	if len(all_words) == 0:
		type_level_oov_rate = 0
		token_level_oov_rate = 0
	else:
		type_level_oov_rate = len(oov_words) / len(all_words)
		token_level_oov_rate = count_oov_words_with_repeated / count_all_words_with_repeated
	return oov_words, count_oov_words_with_repeated, count_all_words_with_repeated, type_level_oov_rate, token_level_oov_rate	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script detects OOV(Out of Vocabulary) words.")
	parser.add_argument("lexicon", type=str, help="path of lexicon.")
	parser.add_argument("text", type=str, nargs='+', help="path text file with kaldi format.")
	parser.add_argument("--force_cased","-f", action="store_true", help="Force text to be the same cased.")
	parser.add_argument("--detail","-d", action="store_true", help="Whether print OOV words.")
	parser.add_argument('--save_in_on', "-s1", action="store_true", help='Whether save OOV word list to one file.')
	parser.add_argument('--save_in_two', "-s2", action="store_true", help='Whether separatly save OOV word list to file.')
	args = parser.parse_args()

	assert os.path.isfile(args.lexicon), f"{args.lexicon} is not exist!"
	for text in args.text:
		assert os.path.isfile(text), f"{text} is not exist!"
	
	word_list = load_lexicon(args.lexicon, args.force_cased)
	oov_words, count_oov_words_with_repeated, count_all_words_with_repeated, type_level_oov_rate, token_level_oov_rate = find_oov_words(args.text, word_list, args.force_cased)

	if len(oov_words) == 0:
		print("No OOV words.")
	else:
		print(f"{count_oov_words_with_repeated}({len(oov_words)}) words of all ({count_all_words_with_repeated}) words not occured in lexicon.")
		print(f"Token level OOV rates (with repeated): {token_level_oov_rate * 100:.4f} %")
		print(f"Type level OOV rates (without repeated): {type_level_oov_rate * 100:.4f} %")
		if args.detail:
			print("OOV words:", oov_words)
		
		if args.save_in_on:
			output_file = os.path.join(os.path.dirname(args.text[0]), "oov_words.txt")
			with open(output_file, 'w', encoding='utf-8') as f:
				for word in oov_words:
					f.write(f"{word}\n")
			print(f"OOV words saved to {args.output}")

		if args.save_in_two:
			# Save Chinese and English OOV words to separate files (one word belongs to one language)
			chinese_file = os.path.join(os.path.dirname(args.text[0]), "oov_words_chinese.txt")
			english_file = os.path.join(os.path.dirname(args.text[0]), "oov_words_english.txt")

			with open(chinese_file, 'w', encoding='utf-8') as cf, \
				 open(english_file, 'w', encoding='utf-8') as ef:
				for w in oov_words:
					if all('\u4e00' <= c <= '\u9fff' for c in w):
						cf.write(f"{w}\n")
					elif all(('a' <= c.lower() <= 'z') or c == "'" for c in w) and w:
						ef.write(f"{w}\n")
					else:
						print(f"Warning: {w} is not a valid Chinese or English word, skipping.")
			print(f"Chinese OOV words saved to {chinese_file}")
			print(f"English OOV words saved to {english_file}")


