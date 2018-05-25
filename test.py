#coding: UTF-8
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import models
import MeCab #分かち書き用

from datetime import datetime
import csv

DATA = []
# refer: https://qiita.com/okadate/items/c36f4eb9506b358fb608
def read_csv(input_file):
	with open(input_file, 'r+') as f:
		reader = csv.reader(f)
		for row in reader:
			# print(row)
			DATA.append(row)

# refer: https://qiita.com/okadate/items/c36f4eb9506b358fb608
def write_csv(output_file):
	with open(output_file, 'w') as f:
		writer = csv.writer(f, lineterminator='\n')
		writer.writerows(DATA)

def do():
	read_csv("./note.csv")
	repl()
	write_csv("./note.csv")
	print(DATA)

def repl():
	while 1:
		#read
		print("コメントを入力してみてください:")
		try:
			comment = input()
		except EOFError:
			print("終了します.")
			break

		date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
		print("[{0}, {1}]".format(comment, date))	
		DATA.append([date, comment])

		#eval
		trainings = []
		for i, data in enumerate(DATA):
			time = data[0]
			comment = data[1]
			# print ("{0},{1}".format(i, data))
			tagger = MeCab.Tagger("-Owakati")
			trainings.append(TaggedDocument(words = tagger.parse(comment).split(),tags = [i]))

		m = models.Doc2Vec(documents=trainings, min_count=1, vector_size=200, sample=3)
		# print (trainings)

		#print
		print("今のコメントに似ているのは...")
		for i in m.docvecs.most_similar(i,topn=3):
			print(DATA[i[0]])

		#loop 
		print("")

if __name__ == '__main__':
	do()


	