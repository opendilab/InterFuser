import os

for i in range(14):
    if not os.path.exists("weather-%d" % i):
        os.mkdir("weather-%d" % i)
    if not os.path.exists("weather-%d/results" % i):
        os.mkdir("weather-%d/results" % i)
