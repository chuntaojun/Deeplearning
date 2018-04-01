import csv


class Write_Csv(object):
    def __init__(self):
        pass

    def write_file(self, predictions, title, id):
        csvFile = open('test_y.csv', 'w')
        write = csv.writer(csvFile)
        write.writerow(title)
        for gender in predictions:
            write.writerow([id, "{:.7f}".format(gender[0])])
            id += 1


if __name__ == '__main__':
    pass
