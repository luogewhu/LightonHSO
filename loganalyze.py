
#!/usr/bin/python
# -*- coding: UTF-8 -*

import os  
list1 = ['android.location.LocationManager.getLastKnownLocation',
		'android.telephony.SmsMessage.createFromPdu',
		'com.android.internal.telephony.RIL.sendSMS',
		'android.telephony.SmsManager.sendTextMessage',
		'android.media.MediaRecorder.start',
		'android.media.AudioRecord.startRecording',
		'android.content.ContentProvider.Transport.query',
		'android.telephony.TelephonyManager.listen',
		'android.telephony.PhoneStateListener.onCallStateChanged',
		'android.content.Intent',
		'android.hardware.camera2.CameraManager.openCamera',
		'android.telephony.TelephonyManager.getDeviceId',
		'android.telephony.TelephonyManager.getSubscriberId',
		'android.telephony.TelephonyManager.getSimSerialNumber',
		'android.telephony.TelephonyManager.getLine1Number'
		 ]
		
path = "F:\LOGS2" 
#G:\dynamicanalyze\benign
#H:\GraduateDesign\111111\logs\new\Android.Backdoor.Spy.a
#F:\LOGS
files= os.listdir(path) 
path2 = "G:\dynamicanalyze\dynamicfeauture4.txt"
output= open(path2,'w')
#s = []  
for file in files: 
     if not os.path.isdir(file): 
		f = open(path+"/"+file)
		iter_f = iter(f)
		#api = [0]  
		api = [1]  
		count = 0
		for line in iter_f: 
			
			if(count<49):
				for i in list1:
					if(line.find(i)!=-1):						
							api.append(list1.index(i)+1)
							count=count+1
							break
             
		
		list2 = []
		for i in range(0,50):
		    list2.append(0)
		api += [0 for i in range(len(list2)-len(api))]
		output.write(str(api)) 
		output.write('\n')
		
		
output.close


						