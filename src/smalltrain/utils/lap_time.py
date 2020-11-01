import time


class LapTime:

    def __init__(self):
        self.time_table = {}
        self.lap_time_table = {}

    def start(self, proc_name):
        try:
            self.lap_time_table[proc_name]['start_time'] = time.time()
        except KeyError:
            self.lap_time_table[proc_name] = {}
            self.start(proc_name)

    def finish(self, proc_name):
        try:
            self.lap_time_table[proc_name]['finish_time'] = time.time()
        except KeyError:
            self.lap_time_table[proc_name] = {}
            self.finish(proc_name)

    def start_time(self, proc_name):
        try:
            return self.lap_time_table[proc_name]['start_time']
        except KeyError:
            return None

    def finish_time(self, proc_name):
        try:
            return self.lap_time_table[proc_name]['finish_time']
        except KeyError:
            return None

    def lap_time(self, proc_name):
        try:
            return self.lap_time_table[proc_name]['finish_time'] - self.lap_time_table[proc_name]['start_time']
        except KeyError:
            return None




