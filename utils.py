import csv


class Logger(object):

    def __init__(self, path, header, mode='w'):
        self.log_file = open(path, mode=mode)
        self.logger = csv.writer(self.log_file, delimiter='\t')

        if mode is not 'a':
            self.logger.writerow(header)

        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()
