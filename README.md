# LightonHSO
Finding Hidden Sensitive Operations in Android APPs 

脚本运行环境 python 2.7 

xiaomiscraper.py--在小米官网爬取APP

loganalyzer.py--分析动态获取的日志，将每个对应的API日志进行提取，如果有日志则相应位为1，存储在一个TXT钟

getFeautures.py--使用androidguard的getpermission进行APK的权限信息提取

0524rocada.py--从动态特征中训练基于决策树的Adaboost,并绘制其ROC曲线

adaboostDCLroc.py--绘制Adaboost的错误率随着迭代次数的关系

svmroc.py--将动态的特征使用SVM分类并绘制ROC曲线
