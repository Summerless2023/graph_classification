import yagmail

print('hello world')
# 登录你的邮箱


yag = yagmail.SMTP(user='xidianzyf@qq.com', password='azgdthbnftwdhbah', host='smtp.qq.com')
# 发送邮件
yag.send(to=['1622496550@qq.com'], subject='主题', contents=['内容', '1231221321'])
