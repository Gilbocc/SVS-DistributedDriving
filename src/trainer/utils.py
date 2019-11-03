import os
import sys

# A helper function to make a directory if it does not exist
def make_dir_if_not_exist(directory):
    if not (os.path.exists(directory)):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# Sets up the logging framework.
def setup_logs(data_dir, experiment_name):
    output_dir = os.path.join(data_dir, 'logs\\{0}\\agent'.format(experiment_name))
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    sys.stdout = open(os.path.join(output_dir, 'out.stdout.txt'), 'w')
    sys.stderr = open(os.path.join(output_dir, 'out.stderr.txt'), 'w')

# Appends a sample to a ring buffer.
# If the appended example takes the size of the buffer over buffer_size, the example at the front will be removed.
def append_to_ring_buffer(item, buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
    buffer.append(item)
    return buffer