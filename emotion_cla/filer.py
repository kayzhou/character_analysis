#coding:utf-8
from emotion_cla import cn_t_2_s
import re
def remove_at(tweet):
    del_at=r'@.*?:|@.*?\s|@.*?$'
    tweet=re.sub(del_at,'',tweet)
    return tweet
def remove_share(tweet):
    tweet=re.sub(r'\(分享自.*?\)','',tweet)
    return  re.sub(r'（分享自.*?）','',tweet)
def remove_emoticon(tweet):
    del_emo=r'\[.*?\]'
    return re.sub(del_emo,'',tweet)
def remove_url(tweet):
    url_pattern=r'http://t.cn/\w+'
    tweet=re.sub(url_pattern,'',tweet)
    return tweet
def filter(weibo):
    weibo=cn_t_2_s.zh_simple(weibo)
    weibo=remove_share(weibo)
    weibo=remove_at(weibo)
#    weibo=remove_emoticon(weibo)
    weibo=remove_url(weibo)
    return weibo
if __name__=='__main__':
    text='【视频[视频]马航MH17客机被击落 机上有295人-视频在线观看-爱奇艺】[视频]马航MH17客机被击落 机上有295人：马航 马航客机 马航客机被击落 马航客机被击毁 马航坠毁 客机坠毁 马航客机... http://t.cn/RPwlyac（分享自 @爱奇艺）'
#    text='远离马航 //【马航客机在乌克兰被击落现场】 (分享自 今日头条)'
    text = '回复@盐城-万象更新:大家都快乐，羊年发羊财。 //@盐城-万象更新:回复@温建宁:谢谢你温教授的提醒，祝福你快乐。 //@温建宁:股市传言纷扰，风雨飘摇之中 //@温建宁:金融股套人的网，已经张开。 在我大力提示下，你是否躲过一劫？'
    text = '心里有事就睡不好，乱乱乱。。。[可怜] http://t.cn/RZTncpi'
    text = '//@全球热门排行榜:太喜欢这个创意了！'
    print(filter(text))
