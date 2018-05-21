# 获取外向和内向用户的id

def get_id(in_name, out_name):
    list_id = []
    for line in open(in_name):
        _id = line.strip().split(' ')[0]
        list_id.append(_id)

    with open(out_name, 'a') as f:
        for _id in list_id:
            f.write(_id + '\n')


get_id('data/large_510_geo_-1.txt', 'uid/uid_-1.txt')
get_id('data/large_510_geo_+1.txt', 'uid/uid_+1.txt')