import time


class Timer:
    """
    A simple timer class to measure the elapsed time of a code block.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.time()
        self.start_process_time = time.process_time()
        self.is_stopped = False
        return self
    
    def stop(self):
        self.end_time = time.time()
        self.end_process_time = time.process_time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_process_time = self.end_process_time - self.start_process_time
        if self.verbose:
            print(f"Elapsed time: {self.elapsed_time:.2f} seconds")
            print(f"Elapsed process time: {self.elapsed_process_time:.2f} seconds")
        self.is_stopped = True
        return self.elapsed_time, self.elapsed_process_time

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.is_stopped:
            self.stop()


