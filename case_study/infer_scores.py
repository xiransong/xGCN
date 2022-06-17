import numpy as np
import sys


def run():
    emb_file = sys.argv[1]
    instance_file = sys.argv[2]
    outfile = sys.argv[3]

    embeddings = np.load(emb_file)
    with open(instance_file, 'r') as rd, open(outfile, 'w') as wt:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split('\t')
            src, dst = int(words[0]), int(words[1])
            score = np.dot(embeddings[src], embeddings[dst])
            wt.write('{0}\t{1}\t{2}\t{3}\n'.format(src, dst, words[2], score))


if __name__ == '__main__':
    run()