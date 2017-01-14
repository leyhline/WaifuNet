# -*- coding: utf-8 -*-
"""
A simple interface for getting files from pCloud.
The files will not be downloaded but instead
returned as Raw class from requests package.

Created on Tue Jan 10 20:26:36 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

import requests
import logging


class PCloud:
    """Class for loading images from PCloud into memory."""
    
    API = "https://api.pcloud.com/"
    
    def __init__(self, username, password):
        """Get access token for further requests."""
        seconds_per_day = 60 * 60 * 24
        seconds_per_month = seconds_per_day * 30
        auth = {"getauth":1, "logout":1,
                "username":username, "password":password,
                "authexpire":seconds_per_month,
                "authinactiveexpire":seconds_per_day}
        r = requests.get(self.API + "userinfo", params=auth)
        self._check_status(r, "Could not get token.")
        self._token = r.json()["auth"]

    def get_files_in_folder(self, *args):
        """Get a list of tuples (filename, fileid) from the given folder structure."""
        root = 0
        for arg in args:
            p = {"auth":self._token, "folderid":root, "nofiles":1}
            r = requests.get(self.API + "listfolder", params=p)
            self._check_status(r, "Visiting folder failed.")
            subfolders = r.json()["metadata"]["contents"]
            fid = (folder["folderid"] for folder in subfolders
                   if folder["isfolder"] and folder["name"] == arg)
            root = next(fid)
        p = {"auth":self._token, "folderid":root}
        r = requests.get(self.API + "listfolder", params=p)
        self._check_status(r, "Could not list files.")
        contents = r.json()["metadata"]["contents"]
        files = [(file["name"], file["fileid"]) for file in contents
                 if not file["isfolder"]]
        return files
        
    def get_file(self, fileid, max_size=5242880):
        """Get a raw file. Maximum bytes retrieved are specified with max_size"""
        p_open = {"auth":self._token, "flags":0, "fileid":fileid}
        s = requests.Session()
        r = s.get(self.API + "file_open", params=p_open)
        self._check_status(r, "Could not get file descriptor for fileid {}.".format(fileid))
        try:
            fd = r.json()["fd"]
        except KeyError as e:
            msg = "Could not get file descriptor for fileid {}.".format(fileid)
            logging.error(msg)
            raise e(msg)
        p_read = {"auth":self._token, "fd":fd, "count":max_size}
        data = s.get(self.API + "file_read", params=p_read, stream=True)
        return data.raw
        
    def _check_status(self, r, error_message=""):
        """Checks if request r was successful (Status: 200) and raises
            exception if necessary with error_message."""
        if r.status_code == 200:
            return True
        else:
            msg = str(r.status_code) + " " + r.reason + " " + error_message
            logging.error(msg)
            raise requests.HTTPError(msg)
