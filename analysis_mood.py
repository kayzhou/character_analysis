# coding: utf8

def get_mood(in_name):
    for line in open(in_name):
        words = line.strip().split(' ')
        uid = words[0]
        clas = words[1]
        m1 = float(words[2])
        m2 = float(words[3])
        m3 = float(words[4])
        m4 = float(words[5])
        m5 = float(words[6])
        _sum = sum([float(m1), float(m2), float(m3), float(m4), float(m5)])
        if _sum != 0:
            print(uid, clas, m1 / _sum, m2 / _sum, m3 / _sum, m4 / _sum, m5 / _sum, file=open('data/large_0225_mood.txt', 'a'), sep=',')


get_mood('data/large_0224_mood.txt')
