#!/usr/bin/python2
import sys
from config import Config

if len(sys.argv) < 3:
	print("Use: ./gen_config.py [config] [path]")
	print("\tconfig\tsmall, medium, large")
	print("\tpath\tOutput path")
	exit() 

config = sys.argv[1]
path = sys.argv[2]

c = Config(config=config, path=path)
c.save()
