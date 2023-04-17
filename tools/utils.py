import os


def snake2pascal(string):
    """Convert Snake case to Pascal case."""
    return string.replace('_', ' ').title().replace(' ', '')

