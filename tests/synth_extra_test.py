"""Tests for the tulip.synth module when gr1c is used."""
import logging
from synth_test import multiple_env_actions_check


logging.getLogger('tulip').setLevel(logging.ERROR)
logging.getLogger('tulip.interfaces.gr1c').setLevel(logging.DEBUG)


def multiple_env_actions_test():
    multiple_env_actions_check('gr1c')
