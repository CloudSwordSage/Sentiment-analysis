import csv

nlpcc2013 = ['./dataset/Nlpcc2013/Nlpcc2013Train.tsv', './dataset/Nlpcc2013/WholeSentence_Nlpcc2013Train.tsv']
nlpcc2013_noNone = ['./dataset/Nlpcc2013/Nlpcc2013Train_NoNone.tsv', './dataset/Nlpcc2013/WholeSentence_Nlpcc2013Train_NoNone.tsv']
nlpcc2014 = ['./dataset/Nlpcc2014/Nlpcc2014Train.tsv', './dataset/Nlpcc2014/WholeSentence_Nlpcc2014Train.tsv']
nlpcc2014_noNone = ['./dataset/Nlpcc2014/Nlpcc2014Train_NoNone.tsv', './dataset/Nlpcc2014/WholeSentence_Nlpcc2014Train_NoNone.tsv']

dataset = nlpcc2014
dataset_noNone = nlpcc2014_noNone

output = './dataset/train.csv'
output_noNone = './dataset/train_noNone.csv'

header = None

with open(output, 'w', newline='', encoding='utf-8') as outfile:
    csv_writer = csv.writer(outfile, delimiter=',')
    for tsv_file in dataset:
        with open(tsv_file, 'r', encoding='utf-8') as infile:
            tsv_reader = csv.reader(infile, delimiter=',')

            if header is None:
                header = next(tsv_reader)
                csv_writer.writerow(header)
            else:
                header = next(tsv_reader)


            for row in tsv_reader:
                csv_writer.writerow(row)

header = None

with open(output_noNone, 'w', newline='', encoding='utf-8') as outfile:
    csv_writer = csv.writer(outfile, delimiter=',')
    for tsv_file in dataset_noNone:
        with open(tsv_file, 'r', encoding='utf-8') as infile:
            tsv_reader = csv.reader(infile, delimiter=',')

            if header is None:
                header = next(tsv_reader)
                csv_writer.writerow(header)
            else:
                header = next(tsv_reader)


            for row in tsv_reader:
                csv_writer.writerow(row)