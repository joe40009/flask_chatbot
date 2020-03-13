from flask import Flask, render_template, request
from config import DevConfig
from keras.models import load_model
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import jieba
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from tensorflow.python.keras.backend import set_session
from rasa_nlu.model import Interpreter

from flask import Flask, render_template, request
from config import DevConfig
import pickle
import jieba
import numpy
import json
import os
import shutil
import pymysql
import datetime
import xmljson
import xml.etree.ElementTree as ET
import requests
import qa_inference2 as qa_inference

interpreter_proj = Interpreter.load("./intent_models/proj2")
interpreter_bill = Interpreter.load("./intent_models/bill2")
interpreter_mb1 = Interpreter.load("./intent_models/mb1")
interpreter_mb2 = Interpreter.load("./intent_models/mb2")
interpreter_mba = Interpreter.load("./intent_models/mba3v1")

sess = tf.Session()
set_session(sess)
graph = tf.get_default_graph()
w2vmodel = Word2Vec.load('./LSTM_model4/aptg_20180309wiki_model.bin')
lstmmodel = load_model('./LSTM_model4/sentiment_test_aptg.h5')
# zzz = np.ones((1, 200))
# lstmmodel.predict(zzz)
# label_dic = {'合約查詢': 0, '帳務查詢': 1, '魔速方塊1.0': 2, '魔速方塊2.0': 3}
label_dic = {'合約查詢': 0, '帳務查詢': 1, '魔速方塊': 2}
gensim_dict = Dictionary()
gensim_dict.doc2bow(w2vmodel.wv.vocab.keys(), allow_update=True)
w2id = {v: k + 1 for k, v in gensim_dict.items()}

app = Flask(__name__)
app.config.from_object(DevConfig)

host="0.0.0.0"
port=36000
qa_inference.init_inference_Engine()
json_file_type ={"cube":"cube_comf.json", "cum":"cum_comf.json" }

entkey = {'PROJ': '方案',
 'CANCEL': '解約',
 'INTERVAL': '合約區間',
 'BRKM': '解約金',
 'BILLCYCLE': '帳單週期',
 'BILLMK': '帳單金額',
 'BILLADDR': '帳寄地址',
 'MAGIC': '魔速方塊',
 'DESC': '魔速方塊2.0介紹',
 'BID': '魔速方塊2.0申辦評估',
 'PERFORMANCE': '魔速方塊1.0效能',
 'DIFF': '魔速方塊2.0與魔速方塊1.0之差異',
 'REPAIR': '魔速方塊2.0報修',
 'CLAIM': '魔速方塊1.0賠償',
 'FREB': '魔速方塊2.0使用頻段',
 'LIMIT': '魔速方塊2.0限定事項',
 'ENV01': '魔速方塊2.0環境限制01',
 'ENV02': '魔速方塊2.0環境限制02',
 'FACTOR': '魔速方塊2.0影響因素'}

def app_commands(_userText, _remote_addr, _if_show=True):
    if _userText == 'deleteall':
        if os.path.isdir('./cum_status/'):
            shutil.rmtree('./cum_status/')
        if _if_show:
            return True, ('已刪除所有使用者資料<br>請重新輸入您的亞太門號')
    if _userText == 'delete':
        if os.path.isdir('./cum_status/' + _remote_addr):
            shutil.rmtree('./cum_status/' + _remote_addr)
        if _if_show:
            return True, ('已刪除此IP使用者資料<br>請重新輸入您的亞太門號')
    return False, _userText
    
def welcome_response(_userText):
    if _userText == '你好':
        return  True, ('您好')
    if '謝謝' in _userText:
        return True, ('很高興為您服務')
    return False, _userText


def dialog_management2(_userText, _remote_addr):
    if _userText =="1":
        app_commands('delete', _remote_addr, _if_show=False)
        return("日後若有需要服務的地方，歡迎您再與我們聯繫。再見")
    elif _userText =="2":
        return("請問您還需要什麼樣的服務?")


def init_json_file(_json_file_type, _remote_addr):
    """
    cube:cube_comf.json\n
    cum:cum_comf.json\n

    """ 
    if _json_file_type=="cum":
        if not os.path.isfile('./cum_status/' + _remote_addr + '/' + json_file_type[_json_file_type]):    
            with open('./cum_status/' + _remote_addr + '/' + json_file_type[_json_file_type], 'w', encoding ="utf-8") as f:
                # json.dump({"name": "null", "phone": "null", "check": "null"}, f, ensure_ascii=False )
                f.write(json.dumps({"name": "null", "phone": "null", "check": "null"},  ensure_ascii=False ))
    elif _json_file_type == "cube":
        if not os.path.isfile('./cum_status/' + _remote_addr + '/' + json_file_type[_json_file_type]):    
            with open('./cum_status/' + _remote_addr + '/' + json_file_type[_json_file_type], 'w', encoding ="utf-8") as f:
                f.write(json.dumps({"name": "null", "phone": "null", "check": "null"},  ensure_ascii=False ))

   

def my_load_json_data(_json_file, _remot_addr):
    data=None
    with open('./cum_status/' + _remot_addr + '/{}'.format(_json_file), encoding ="utf-8") as jf:
        data = json.load(jf)
    return data




@app.route('/')
def index():
    return render_template("./index.html")
    # return "text"

@app.route("/get")
#function for the bot response
def get_bot_response(_debug=False, _demoText=''):
    if _debug:
        remote_addr ="0.0.0.0"
    else:
        remote_addr=request.remote_addr


    if not os.path.isdir('./cum_status/' + remote_addr):
        os.makedirs('./cum_status/' + remote_addr)
    
    check_list = os.listdir('./cum_status/' + remote_addr)

    if _debug:
        userText =_demoText
    else:
        userText = request.args.get('msg')
        
    app_command_flag = False
    app_command_flag, app_txt = app_commands(userText, remote_addr)
    if app_command_flag:
        return app_txt
    
    new_sen = userText
    new_sen_list = jieba.lcut(new_sen)
    sen2id = [w2id.get(word,0) for word in new_sen_list]
    sen_input = pad_sequences([sen2id], maxlen=200)
    global lstmmodel
    global sess
    with graph.as_default():
        set_session(sess)
        res = lstmmodel.predict(sen_input)[0]
    dompb = str(res[np.argmax(res)])
    if not '魔速方塊' in userText:
        if res[np.argmax(res)] > 0.8:
            result = list(label_dic.keys())[list(label_dic.values()).index(np.argmax(res))]
        #         return ("<font color='gray'>Probability：{}，Domain：{}".format(str(res[np.argmax(res)]), result) + "</font><br>")
        else:
            result = list(label_dic.keys())[list(label_dic.values()).index(np.argmax(res))]
        #         if not res[np.argmax(res)] > 0.5 and result == '魔速方塊':
        #         if not '魔速方塊' in userText:
            return ("<font color='gray'>Probability：{}，Domain：{}".format(str(res[np.argmax(res)]), result) + "</font><br>" + \
                        "我不了解您的問題，請專人為您服務")
    else:
        result = '魔速方塊'
    
#     result = '魔速方塊'
    
    if os.path.isfile('./cum_status/' + remote_addr + '/cube_ex.json'):
        os.remove('./cum_status/' + remote_addr + '/cube_ex.json')
        result = '魔速方塊'    
    
    
        # 合約查詢 前導      
    # if not os.path.isfile('./cum_status/' + remote_addr + '/cum_comf.json'):    
    #     with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w', encoding ="utf-8") as f:
    #         json.dump({"name": "null", "phone": "null", "check": "null"}, f, )
    init_json_file('cum', remote_addr)

    # with open('./cum_status/' + remote_addr + '/cum_comf.json', encoding ="utf-8") as jf:
    #     data = json.load(jf)
    data = my_load_json_data('cum_comf.json', remote_addr)
    
    # client身分資料 check
    if data['name'] == 'null' or data['check'] == 'null' or data['phone'] == 'null':
        
        # with open('./cum_status/' + remote_addr + '/cum_comf.json', encoding ="utf-8") as jf:
        #     data = json.load(jf)
        data = my_load_json_data('cum_comf.json', remote_addr)
            
        if data['phone'] == 'null':            
            try:
                request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                        '?command=GetContractBasicInfo&MDN=' + userText)
                xml = ET.fromstring(request_data.text)
                crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
                phone_data = crmjs['cws-api']['CustomerCName']
                # sql = "select CustomerCName FROM GetContractBasicInfo_api where MDN = " + userText
                # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                # cursor = db.cursor()
                # cursor.execute(sql)
                # phone_data = cursor.fetchall()

                if phone_data == {}:
                    return ('查無此號碼，請重新輸入電話')
                else:            
                    data['phone'] = userText
                    with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w', encoding ="utf-8") as f:
                        # json.dump(data, f, indent=4, ensure_ascii=False )
                        f.write(json.dumps(data, ensure_ascii=False ))
                    return ('請輸入姓名')
            except:
                return ('請輸入電話')
        
        else:
            if data['name'] == 'null':
                request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                        '?command=GetContractBasicInfo&MDN=' + data['phone'])
                xml = ET.fromstring(request_data.text)
                crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
                cum_data = crmjs['cws-api']['CustomerCName']['$']                
                # sql = "select CustomerCName FROM GetContractBasicInfo_api where MDN = " + data['phone']
                # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                # cursor = db.cursor()
                # cursor.execute(sql)
                # cum_data = cursor.fetchall()                
                # if cum_data[0][0] == userText:
                if cum_data == userText:
                    data['name'] = userText
                    with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w', encoding ="utf-8") as f:
                        # json.dump(data, f,indent=4, ensure_ascii=False )
                        # json.dump({"name": "null", "phone": "null", "check": "null"}, f)
                        f.write(json.dumps(data, ensure_ascii=False))
                    return ('是否本人')
                else:
                    with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w',  encoding ="utf-8") as f:
                        # json.dump({"name": "null", "phone": "null", "check": "null"}, f,ensure_ascii=False )
                        f.write(json.dumps({"name": "null", "phone": "null", "check": "null"},  ensure_ascii=False ))
                    return ('電話與姓名不符，請重新輸入電話')
                    
            else:
                if data['check'] == 'null':
                    if '否' in userText or '不' in userText or '非' in userText or userText.lower() == 'no':
                        data['check'] = 0                        
                        with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w', encoding='utf-8') as f:
                            # json.dump(data, f, ensure_ascii=False )
                            f.write(json.dumps(data, ensure_ascii=False))
                            # json.dumps(data,  ensure_ascii=False )
                        return ('不是本人的話，有部分權限無法操作，請問要提供甚麼服務嗎?')
                    
                    elif '本人' in userText or '是' in userText  or '對' in userText or '確認' in userText or userText.lower() == 'yes' or '沒錯' in userText:
                        data['check'] = 1
                        
                        request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                        '?command=GetContractBasicInfo&MDN=' + data['phone'])
                        xml = ET.fromstring(request_data.text)
                        crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
                        # crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml),indent=4, ensure_ascii=False ))
                        CustomerID_ck_results = crmjs['cws-api']['CustomerID']['$']
                        data['custid'] = CustomerID_ck_results
                        # sql = "select CustomerID FROM GetContractBasicInfo_api where MDN = " + data['phone']
                        # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                        # cursor = db.cursor()
                        # cursor.execute(sql)
                        # CustomerID_ck_results = cursor.fetchall()
                        # data['custid'] = CustomerID_ck_results[0][0]
                        with open('./cum_status/' + remote_addr + '/cum_comf.json', 'w', encoding='utf-8') as f:
                            # json.dump(data, f,indent=4, ensure_ascii=False )
                            f.write(json.dumps(data, ensure_ascii=False))
                            
                            return ('確認身份，' + data['name'] + ' 先生/小姐您好，請問要提供甚麼服務嗎?')
                    else:
                        return ('請確認是否是本人')
    
    
        ##################################
    # 抱歉對應
    if os.path.isfile('./cum_status/' + remote_addr + '/fire_why.txt'):
        with open('./cum_status/' + remote_addr + '/fire_why.txt', 'w', encoding='utf-8') as f:
            f.write(userText)
        if os.path.isfile('./cum_status/' + remote_addr + '/fire_why_ck.txt'):
            os.remove('./cum_status/' + remote_addr + '/fire_why_ck.txt')
        os.rename('./cum_status/' + remote_addr + '/fire_why.txt', './cum_status/' + remote_addr + '/fire_why_ck.txt')
        return ('很抱歉造成您不好的體驗，請問是否還有其他可以為您服務的地方呢？')
    
    
    if result == '合約查詢':
        result_int = interpreter_proj.parse(userText)
        
        if result_int['intent']['name'] == 'PROJ':
            data = my_load_json_data('cum_comf.json', remote_addr)

            phone = data['phone']
            # sql = "select RatePlanIdDesc FROM GetContractBasicInfo_api where MDN = " + phone
            # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
            # cursor = db.cursor()
            # cursor.execute(sql)
            # cum_data = cursor.fetchall()
            request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                    '?command=GetContractBasicInfo&MDN=' + phone)
            xml = ET.fromstring(request_data.text)
            crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
            cum_data = crmjs['cws-api']['RatePlanIdDesc']['$']

            # return ('您的門號目前是搭配' + cum_data[0][0])
            return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                    '您的門號目前是搭配' + cum_data)
        
        if result_int['intent']['name'] == 'INTERVAL':
            data = my_load_json_data('cum_comf.json', remote_addr)

            phone = data['phone'] 
            request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                    '?command=GetContractBasicInfo&MDN=' + phone)
            xml = ET.fromstring(request_data.text)
            crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False))
            if type(crmjs['cws-api']['PromoInfo']) == dict:
                act = crmjs['cws-api']['PromoInfo']['ActualStartDate']['$']
                exp = crmjs['cws-api']['PromoInfo']['ExpireDate']['$']
            else:
                for prom in crmjs['cws-api']['PromoInfo']:    
                    act = prom['ActualStartDate']['$']
                    exp = prom['ExpireDate']['$']    
                    if datetime.datetime.strptime(act, "%Y/%m/%d") < datetime.datetime.now() + datetime.timedelta(hours=8) < datetime.datetime.strptime(exp, "%Y/%m/%d"):
                        break
            return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + '您的門號合約日期' + act + '~' + exp)
        
        if result_int['intent']['name'] == 'BRKM':
            data = my_load_json_data('cum_comf.json', remote_addr)

            phone = data['phone']
            # sql = "select BreakMoney FROM GetContractBasicInfo_api where MDN = " + phone
            # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
            # cursor = db.cursor()
            # cursor.execute(sql)
            # cum_data = cursor.fetchall()
            request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                    '?command=GetContractBasicInfo&MDN=' + phone)
            xml = ET.fromstring(request_data.text)
            crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
            if type(crmjs['cws-api']['PromoInfo']) == dict:
                act = crmjs['cws-api']['PromoInfo']['ActualStartDate']['$']
                exp = crmjs['cws-api']['PromoInfo']['ExpireDate']['$']
                try:
                    BreakMoney = crmjs['cws-api']['PromoInfo']['BreakMoney']['$']
                except:
                    BreakMoney = 0
            else:
                for prom in crmjs['cws-api']['PromoInfo']:    
                    act = prom['ActualStartDate']['$']
                    exp = prom['ExpireDate']['$']
                    try:
                        BreakMoney = prom['BreakMoney']['$']
                    except:
                        BreakMoney = 0
                    if datetime.datetime.strptime(act, "%Y/%m/%d") < datetime.datetime.now() + datetime.timedelta(hours=8) < datetime.datetime.strptime(exp, "%Y/%m/%d"):
                        break
            # if cum_data[0][0] == 0:
            if BreakMoney == {} or int(BreakMoney) <= 0:
                return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                        '您的門號沒有專案補貼款(解約金)')

            return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                    '您的門號專案補貼款(解約金)目前是' + str(BreakMoney) + '元')
        
        if result_int['intent']['name'] == 'CANCEL':
            data = my_load_json_data('cum_comf.json', remote_addr)
            phone = data['phone']
            # sql = "select RatePlanIdDesc, ActDate, ExpireDate, BreakMoney FROM GetContractBasicInfo_api where MDN = " + phone
            # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
            # cursor = db.cursor()
            # cursor.execute(sql)
            # cum_data = cursor.fetchall()
            request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                    '?command=GetContractBasicInfo&MDN=' + phone)
            xml = ET.fromstring(request_data.text)
            crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml) ))
            if type(crmjs['cws-api']['PromoInfo']) == dict:
                act = crmjs['cws-api']['PromoInfo']['ActualStartDate']['$']
                exp = crmjs['cws-api']['PromoInfo']['ExpireDate']['$']
                try:
                    BreakMoney = crmjs['cws-api']['PromoInfo']['BreakMoney']['$']
                except:
                    BreakMoney = 0
            else:
                for prom in crmjs['cws-api']['PromoInfo']:    
                    act = prom['ActualStartDate']['$']
                    exp = prom['ExpireDate']['$']
                    try:
                        BreakMoney = prom['BreakMoney']['$']
                    except:
                        BreakMoney = 0
                    if datetime.datetime.strptime(act, "%Y/%m/%d") < datetime.datetime.now() + datetime.timedelta(hours=8) < datetime.datetime.strptime(exp, "%Y/%m/%d"):
                        break
            RPID = crmjs['cws-api']['RatePlanIdDesc']['$']            

            with open('./cum_status/' + remote_addr + '/fire_why.txt', 'w', encoding ="utf-8") as f:        

                if int(BreakMoney) <= 0:
                    return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                            '您的合約是' + RPID + '，合約時間是' + act + '~' + exp + \
                            '若要解約的話，目前無專案補貼款(解約金)。'+ \
                            '若您確定要解約的話，需有勞持雙證件至亞太直營或加盟門市辦理。' + '<br/>' + 
                            '冒昧詢問，您想要解約的原因是?')
                else:
                    return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                            '您的合約是' + RPID + '，合約時間是' + act + '~' + exp + \
                            '若要解約的話，專案補貼款(解約金)目前是' + str(BreakMoney) + '元。' + \
                            '若您確定要解約的話，需有勞持雙證件至亞太直營或加盟門市辦理。' + '<br/>' + 
                            '冒昧詢問，您想要解約的原因是?')

        
        
    if result == '帳務查詢':
        result_int = interpreter_bill.parse(userText)
        
        if result_int['intent']['name'] == 'BILLMK':
            data = my_load_json_data('cum_comf.json', remote_addr)
            phone = data['phone']
            custid = data['custid']
            # sql = "select Pbalance, Duedate FROM APBWPrintBill_api where Custcode = " + "'" + custid + "'"            
            # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
            # cursor = db.cursor()
            # cursor.execute(sql)
            # bill_data = cursor.fetchall()
            request_data = requests.get('http://10.1.11.37:8080/brm-http/services/APBWPrintBill?' + \
                                        'Command=APBWPrintBill&custcode=' + custid + '&mdn=' + phone)
            xml = ET.fromstring(request_data.text)
            billjs = json.loads(json.dumps(xmljson.badgerfish.data(xml) ))
            billmoney = billjs['fnz-api']['InvoiceList']['Current_amt']['$']
            billtime = billjs['fnz-api']['InvoiceList']['Duedate']['$']
            # if bill_data[0][0] == '0':
            if billmoney <= 0:
                return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                        '目前無需繳款')
            else:
                # return ('尚須繳款金額' + bill_data[0][0] + '，繳款截止日' + bill_data[0][1] + '前')
                return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                        '尚須繳款金額' + str(billmoney) + '元，繳款截止日' + billtime + '前')
        
        if result_int['intent']['name'] == 'BILLCYCLE':
            data = my_load_json_data('cum_comf.json', remote_addr)

            phone = data['phone']
            custid = data['custid']
            # sql2 = "select PayPeriod FROM GetContractBasicInfo_api where CustomerID = " + "'" + custid + "'"
            # db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
            # cursor = db.cursor()
            # cursor.execute(sql2)
            # cum_data = cursor.fetchall()            
            request_data = requests.get('http://crm00.apt.corp/cmsweb/CWS/GetContractBasicInfo.jsp' + \
                                    '?command=GetContractBasicInfo&MDN=' + phone)
            xml = ET.fromstring(request_data.text)
            crmjs = json.loads(json.dumps(xmljson.badgerfish.data(xml), ensure_ascii=False ))
            cum_data = crmjs['cws-api']['PayPeriod']['$']
            if cum_data == 5:
                billtime = '每月5至次月4日'
                getbill = '每月16日到20日'
            if cum_data == 10:
                billtime = '每月10至次月9日'
                getbill = '每月21日到25日'
            if cum_data == 15:
                billtime = '每月15至次月14日'
                getbill = '每月26日到30日'
            if cum_data == 20:
                billtime = '每月20至次月19日'
                getbill = '每月30日到5日'           
            return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + \
                    '帳單計算週期是' + billtime + '，預計收到帳單日期為' + getbill)
    
        
    if result == '魔速方塊':
        if not os.path.isdir('./cum_status/' + remote_addr):
            os.makedirs('./cum_status/' + remote_addr)

        userText = request.args.get('msg')
        userText = userText.replace(" ","")

        if str(1) not in userText and str(2) not in userText and not os.path.isfile('./cum_status/' + remote_addr + '/cube_ver.json') and not os.path.isfile('./cum_status/' + remote_addr + '/cube_comf.json'):        
            with open('./cum_status/' + remote_addr + '/cube_tmp.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps({"text": userText}, ensure_ascii=False))
            with open('./cum_status/' + remote_addr + '/cube_ex.json', 'w', encoding='utf-8') as f:
                f.write('')
            return ('請問您的魔速方塊是1.0還是2.0!!')
        else:
            if str(1) in userText:
                cube_ver = 1
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps({"ver": 1}, ensure_ascii=False))
            elif str(2) in userText:
                cube_ver = 2
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps({"ver": 2}, ensure_ascii=False))


        with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
            cube_ver = json.load(f)["ver"]
            
            
        if not os.path.isfile('./cum_status/' + remote_addr + '/cube_tmp.json'):
            if userText == '申辦' or userText == '1':
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                if cube_ver == 1:
                    qatext = "魔速方塊1.0申辦評估"
                elif cube_ver == 2:
                    qatext = "魔速方塊2.0申辦評估"
                sql = "select content from Fact_content_Serving where title_nm = " + "'" + qatext + "'"
                db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                cursor = db.cursor()
                cursor.execute(sql)
                mb_data = cursor.fetchall()
                my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = str(userText), _doc_tokens = mb_data[0][0])
                if my_result['probability'] > 0.8:
                    return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                            "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                            "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                            "<font color='gray'>選擇文本名稱："+ qatext + '</font><br/>' + 
                            my_result["text"].replace(" ",""))
                else:
                    if cube_ver == 1:
                        return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                                "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                                "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                                '只要是亞太電信的4G月租型有效用戶且家中有ADSL/寬頻網路 即可申裝魔速方塊1.0，但須視亞太電信到場評估結果而定。')
                    elif cube_ver == 2:
                        return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                                "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                                "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                                '魔速方塊2.0的申辦對象為4G月租型有效用戶，申辦不用付費，但需由亞太電信的工程師到現場評估後才能決定是否適合安裝。')

            if userText == '功能' or userText == '2':
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                if cube_ver == 1:
                    qatext = "魔速方塊1.0功能"
                elif cube_ver == 2:
                    qatext = "魔速方塊2.0功能"
                sql = "select content from Fact_content_Serving where title_nm = " + "'" + qatext + "'"
                db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                cursor = db.cursor()
                cursor.execute(sql)
                mb_data = cursor.fetchall()
                my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = str(userText), _doc_tokens = mb_data[0][0])
                if my_result['probability'] > 0.8:
                    return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                            "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                            "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                            "<font color='gray'>選擇文本名稱："+ qatext + '</font><br/>' + 
                            my_result["text"].replace(" ",""))
                else:
                    if cube_ver == 1:
                        return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                                "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                                "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                                '魔速方塊1.0主要是提供優化用戶室內涵蓋品質，涵蓋範圍約半徑30米左右，若想要體驗高速數據服務，則建議固網寬頻要有50MB以上。')
                    elif cube_ver == 2:
                        return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                                "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                                "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                                '魔速方塊2.0的功能在於延伸基站涵蓋的範圍，主要解決基站邊緣收訊不良問題，增強用戶室內信號，室內的涵蓋範圍約50公尺。魔速方塊2.0的頻率不會跟WiFi訊號互相干擾。')  

            if userText == '與Wi-Fi通話之區別' or (userText == '3' and cube_ver == 1):
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        "魔速方塊1.0和Wi-Fi通話不同之處在於魔速方塊使用的是亞太4G專用頻段。Wi-Fi通話是藉由無線介接技術讓用戶能在有Wi-Fi的環境下使用行動業者提供的上網及語音服務。")

            if userText == '魔速方塊1.0與2.0之比較' or (userText == '4' and cube_ver == 1):
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        '在室內涵蓋上，魔速方塊1.0約30公尺，魔速方塊2.0約50公尺。在發射功率上，魔速方塊1.0為100mW，魔速方塊2.0為150mW。在安裝位置上，魔速方塊1.0可裝在地下室或高樓層，魔速方塊2.0必須安裝於可接收到外部訊號的窗邊。頻段的差別，魔速方塊1.0為700MHz或2600MHz，魔速方塊2.0為2600MHz。最後，魔速方塊1.0需要接有線固網並建議該固網要有50M以上，魔速方塊2.0不用接固網')

            if userText == '室內安裝' or (userText == '3' and cube_ver == 2):
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                qatext = "魔速方塊2.0收訊環境"
                sql = "select content from Fact_content_Serving where title_nm = " + "'" + qatext + "'"
                db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                cursor = db.cursor()
                cursor.execute(sql)
                mb_data = cursor.fetchall()
                my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = str(userText), _doc_tokens = mb_data[0][0])
                if my_result['probability'] > 0.8:
                    return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                            "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                            "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                            "<font color='gray'>選擇文本名稱："+ qatext + '</font><br/>' + 
                            my_result["text"].replace(" ",""))
                else:
                    return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                            "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                            "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                            '原則上只要在室內可接收到基站訊號，儘可能安裝於室內陽台、窗台、窗沿、窗邊桌子。即可透過魔速方塊2.0增強室內收訊，而隔著牆壁確實有信號上衰減問題。')

            if userText == '可支援的手機' or (userText == '4' and cube_ver == 2):
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        '目前大多數較新的智慧型手機都能正常工作，只需確認終端需支持2600MHz B38規格。而使用VoLTE手機可以得到較佳通話品質。')

            if userText =='魔速方塊2.0與1.0之比較' or (userText == '5' and cube_ver == 2):
                with open('./cum_status/' + remote_addr + '/cube_comf.json', 'r', encoding='utf-8') as f:
                    userText = json.load(f)["text"]
                os.remove('./cum_status/' + remote_addr + '/cube_comf.json')
                with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                    cube_ver = json.load(f)["ver"]
                return ("<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        '魔速方塊2.0與1.0的區別如下，在室內涵蓋比較上，魔速方塊1.0的室內涵蓋大約30公尺，魔速方塊2.0室內涵蓋大約50公尺。在發射功率比較上，魔速方塊1.0為100mW，魔速方塊2.0為150mW。在安裝位置差異上，魔速方塊1.0可裝在地下室或高樓層，魔速方塊2.0必須安裝於可接收到外部訊號的窗邊。在輪出頻段的差別，魔速方塊1.0為700MHz或2600MHz，魔速方塊2.0為2600MHz。魔速方塊1.0要接有線固網並建議該固網要有50Mbps以上，魔速方塊2.0不用接固網。')


        if os.path.isfile('./cum_status/' + remote_addr + '/cube_tmp.json'):
            with open('./cum_status/' + remote_addr + '/cube_tmp.json', 'r', encoding='utf-8') as f:
                userText = json.load(f)["text"]
            os.remove('./cum_status/' + remote_addr + '/cube_tmp.json')

        result_int = interpreter_mba.parse(userText)


        if result_int['intent']['confidence'] > 0.8:

            with open('./cum_status/' + remote_addr + '/cube_ver.json', 'r', encoding='utf-8') as f:
                cube_ver = json.load(f)["ver"]

            if cube_ver == 1:
                entl = []
                for ent in result_int['entities']:
                    entl.append(ent['entity'])

                if str(1) in userText and str(2) in userText:
                    qatext = '魔速方塊2.0與1.0之比較'                    
                elif 'CLAIM' in entl or result_int['intent']['name'] == 'CLAIM':
                    qatext = '魔速方塊1.0賠償' 
                elif 'ENV' in entl and 'SIGNAL' in entl:
                    qatext = '魔速方塊1.0功能'    
                elif 'SIGNAL' in entl:
                    qatext = '魔速方塊1.0介紹'
                elif 'STATUS' in entl:
                    qatext = '魔速方塊1.0賠償' 
                elif 'BID' in entl:
                    qatext = '魔速方塊1.0申辦評估'
                elif 'KPI' in entl or (entl == [] and result_int['intent']['name'] == 'KPI'):
                    qatext = '魔速方塊1.0功能'
                elif 'NETWORK' in entl or (entl == [] and result_int['intent']['name'] == 'NETWORK'):
                    qatext = '魔速方塊1.0功能'
                else:
                    qatext = '魔速方塊1.0與2.0之比較'


                sql = "select content from Fact_content_Serving where title_nm = " + "'" + qatext + "'"
                db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                cursor = db.cursor()
                cursor.execute(sql)
                mb_data = cursor.fetchall()
            #         try:
                my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = str(userText), _doc_tokens = mb_data[0][0])
                return ("<font color='gray'>confidence：{}".format(str(result_int['intent']['confidence'])) + "</font><br>" + 
                        "<font color='gray'>intent：{}".format(str(result_int['intent']['name'])) + "</font><br>" + 
                        "<font color='gray'>entity：{}".format(str([re['value'] for re in result_int['entities']])) + "</font><br>" + 
                        "<font color='gray'>ner：{}".format(str(entl)) + "</font><br>" + 
                        "<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" +
                        "<font color='gray'>選擇文本名稱："+ qatext + '</font><br/>' + my_result['text'].replace(" ",""))


            if cube_ver == 2:
                entl = []
                for ent in result_int['entities']:
                    entl.append(ent['entity'])

                if str(1) in userText and str(2) in userText:
                    qatext = '魔速方塊2.0與1.0之比較'
                elif 'SIGNAL' in entl and 'STATUS' in entl:
                    qatext = '魔速方塊2.0收訊環境'
                elif 'DEVICE' in entl and 'ENV' in entl:
                    qatext = '魔速方塊2.0可支援手機'
                elif 'KPI' in entl and 'NETWORK' in entl:
                    qatext = '魔速方塊2.0可支援手機'
                elif 'ENV' in entl and 'KPI' in entl:
                    qatext = '魔速方塊2.0功能'
                elif 'AMT' in entl:
                    qatext = '魔速方塊2.0用電'
                elif 'NETWORK' in entl:
                    qatext = '魔速方塊2.0功能'
                elif 'STATUS' in entl:
                    qatext = '魔速方塊2.0賠償'
                elif 'SIGNAL' in entl:
                    qatext = '魔速方塊2.0功能'
                elif 'ENV' in entl and 'KPI' in entl:
                    qatext = '魔速方塊2.0功能'
                elif 'BID' in entl:
                    qatext = '魔速方塊2.0申辦評估'
                elif 'KPI' in entl or (entl == [] and result_int['intent']['name'] == 'KPI'):
                    qatext = '魔速方塊2.0電磁波'
                elif 'DEVICE' in entl:
                    qatext = '魔速方塊2.0收訊環境'
                elif 'ENV' in entl:
                    qatext = '魔速方塊2.0收訊環境'
                elif 'CLAIM' in entl or (entl == [] and result_int['intent']['name'] == 'CLAIM'):
                    qatext = '魔速方塊2.0賠償'
                else:
                    qatext = '魔速方塊2.0與1.0之比較'

                sql = "select content from Fact_content_Serving where title_nm = " + "'" + qatext + "'"
                db = pymysql.connect(host='172.16.56.101', port=31996, user='root', passwd='password', db='textdb' )
                cursor = db.cursor()
                cursor.execute(sql)
                mb_data = cursor.fetchall()
            #         try:
                my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = str(userText), _doc_tokens = mb_data[0][0])
                return ("<font color='gray'>confidence：{}".format(str(result_int['intent']['confidence'])) + "</font><br>" + 
                        "<font color='gray'>intent：{}".format(str(result_int['intent']['name'])) + "</font><br>" + 
                        "<font color='gray'>entity：{}".format(str([re['value'] for re in result_int['entities']])) + "</font><br>" + 
                        "<font color='gray'>ner：{}".format(str(entl)) + "</font><br>" + 
                        "<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        "<font color='gray'>question：{}".format(str(userText)) + "</font><br>" + 
                        "<font color='gray'>probability：{}".format(my_result['probability']) + "</font><br>" + 
                        "<font color='gray'>選擇文本名稱："+ qatext + '</font><br/>' + my_result['text'].replace(" ",""))

        else:
            with open('./cum_status/' + remote_addr + '/cube_ex.json', 'w', encoding='utf-8') as f:
                f.write('')
            
            entl = []
            for ent in result_int['entities']:
                entl.append(ent['entity'])

            with open('./cum_status/' + remote_addr + '/cube_comf.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps({"text": userText}, ensure_ascii=False))
            if cube_ver == 1:
                return ("<font color='gray'>confidence：{}".format(str(result_int['intent']['confidence'])) + "</font><br>" + 
                        "<font color='gray'>intent：{}".format(str(result_int['intent']['name'])) + "</font><br>" + 
                        "<font color='gray'>entity：{}".format(str([re['value'] for re in result_int['entities']])) + "</font><br>" + 
                        "<font color='gray'>ner：{}".format(str(entl)) + "</font><br>" + 
                        "<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        '請問您要詢問：<br>(1)申辦<br>(2)功能<br>(3)與Wi-Fi通話之區別<br>(4)魔速方塊1.0與2.0之比較<br>')
            elif cube_ver == 2:
                return ("<font color='gray'>confidence：{}".format(str(result_int['intent']['confidence'])) + "</font><br>" + 
                        "<font color='gray'>intent：{}".format(str(result_int['intent']['name'])) + "</font><br>" + 
                        "<font color='gray'>entity：{}".format(str([re['value'] for re in result_int['entities']])) + "</font><br>" + 
                        "<font color='gray'>ner：{}".format(str(entl)) + "</font><br>" + 
                        "<font color='gray'>cube_ver：{}".format(str(cube_ver)) + "</font><br>" + 
                        '請問您要詢問：<br>(1)申辦<br>(2)功能<br>(3)室內安裝<br>(4)可支援的手機<br>(5)魔速方塊2.0與1.0之比較')        

    
    return ("<font color='gray'>Probability：{}，Domain：{}，Intent：{}".format(dompb, result, entkey[result_int['intent']['name']]) + "</font><br>" + "我不了解您的問題，請專人為您服務")



if __name__ == 'main':
    app.run(host,port)

