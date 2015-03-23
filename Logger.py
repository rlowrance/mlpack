import sys


class Logger(object):
    # from stack overflow: how do i duplicat sys stdout to a log file in python

    def __init__(self, logfile_path, logfile_mode='w'):
        self.terminal = sys.stdout
        self.log = open(logfile_path, logfile_mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

# sys.stdout = Logger()  # now the print statement echos to screen and log file
