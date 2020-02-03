import os
import logging

logger = logging.getLogger(__name__)


def file_list(file_args):
    comma_sep = file_args.split(',')
    files = []
    for csep in comma_sep:
        if os.path.isdir(csep):
            files.extend([os.path.join(csep, f) for f in os.listdir(csep)])
        elif os.path.exists(csep):
            files.append(csep)
        else:
            raise ValueError(f"Not found: {csep}")
    return files

