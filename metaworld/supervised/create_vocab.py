import pickle
import string
translator = str.maketrans('', '', string.punctuation)

def main():
	vocab = []
	with open('train.txt') as f:
		for line in f.readlines():
			line = line.strip()
			words = line.split('\t')[-1].split()
			words = [w.translate(translator).lower() for w in words]
			vocab += words
	with open('valid.txt') as f:
		for line in f.readlines():
			line = line.strip()
			words = line.split('\t')[-1].split()
			words = [w.translate(translator).lower() for w in words]
			vocab += words
	vocab = list(set(vocab))
	print(vocab)
	print(len(vocab))
	pickle.dump(vocab, open('vocab.pkl', 'wb'))

if __name__ == '__main__':
	main()