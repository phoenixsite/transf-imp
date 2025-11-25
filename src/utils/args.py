import argparse

def positive_number(string):
    """
    Convert a string to a positive number.

    Raise an ArgumentError if ``string`` represents a negative or zero
    number.
    """

    number = int(string)

    if number <= 0:
        raise argparse.ArgumentTypeError("it must be a positive number.")
    return number
