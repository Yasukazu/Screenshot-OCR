import logging, sys

stdout_handler = logging.StreamHandler(stream=sys.stdout)
format_output = logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s') # <-
stdout_handler.setFormatter(format_output)