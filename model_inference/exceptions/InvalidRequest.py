#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Author Name"""
__author__ = "Ivan de Paz Centeno"

class InvalidRequest(Exception):
    """
    This Class Handles Invalid Requests
    """
    def __init__(self, message, status_code=400, payload=None):
        
        """
        initialization for the exception class
        :param message:  exception message
        :param status_code:  exception code
        :param payload: exception payload
        """

        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        
        """
        function to convert the exception into a dictionary
        :return: the exception dit
        """
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv
