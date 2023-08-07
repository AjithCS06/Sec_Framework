from django.shortcuts import render
import json
from django.core import serializers
from .models import *
from .validation import *
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.db.models import Count
from django.views.decorators.cache import never_cache
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from django.conf import settings
import datetime
from tinyec import registry #('tinyec library' for ECDH in Python )
import secrets
import re
import os
import ipfshttpclient
from .c1 import *
import pandas as pd
import numpy as np
import joblib

# Create your views here.


@never_cache
def show_index(request):
	return render(request, "login.html", {})


@never_cache
def logout(request):
	if 'uid' in request.session:
		del request.session['uid']
	return render(request,'login.html')




def check_login(request):
	username = request.POST.get("username")
	password = request.POST.get("password")

	print(username)
	print(password)

	res=validate(username,password,request)

	if res=="invalid":
		return HttpResponse("<script>alert('Invalid');window.location.href='/show_index/'</script>")
	else:
		return HttpResponse("<script>alert('Login Successful');window.location.href='/show_home_user/'</script>")


@never_cache
###############ADMIN START
def show_home_user(request):
	if 'uid' in request.session:
		print(request.session['uid'])
		uname=request.session['uid']
		return render(request,'home_user.html',{"uname":uname}) 
	else:
		return render(request,'login.html')


def generate_folder(name):
	folder_path = 'enc_app/static/Datas/'+name
	os.makedirs(folder_path, exist_ok=True)
	return folder_path


@never_cache
def upload_file(request):
	if 'uid' in request.session:

		return render(request,'upload_file.html') 
	else:
		return render(request,'login.html')


def compress(pubKey):
	return hex(pubKey.x) + hex(pubKey.y % 2)[2:]
loc='enc_app/static/Datas/'
def generate_key():
	#The elliptic curve used for the ECDH calculations is 256-bit named curve brainpoolP256r1
	curve = registry.get_curve('brainpoolP256r1')

	PrivKey = secrets.randbelow(curve.field.n)
	print("private key:", PrivKey)
	PubKey = PrivKey * curve.g
	my_pubkey=compress(PubKey)
	print("public key:", compress(PubKey))

	return my_pubkey,PrivKey


def create_folder():
	folder_path = 'enc_app/static/Uploaded_Data'
	os.makedirs(folder_path, exist_ok=True)
	return folder_path


def store_key(uname,p_key1,p_key2):
	folder_path = os.path.join('enc_app/static/User_Keys', uname)
	os.makedirs(folder_path, exist_ok=True)

	# Create a text file and store the p_key in it
	file_path = os.path.join(folder_path, 'key.txt')
	with open(file_path, 'w') as file:
		file.write(str(p_key1))
		file.write("\n")
		file.write(str(p_key2))
	return folder_path


def upload(request):
	file1 = request.FILES["folder"]
	file_name=file1.name
	uname=request.session["uid"]

	folder_path = create_folder()
	folder_path1=generate_folder(uname)
	fs = FileSystemStorage(folder_path)
	fs.save(file_name, file1)
	fs1 = FileSystemStorage(folder_path1)
	fs1.save(file_name, file1)

	p_key1,p_key2=generate_key()

	var=store_key(uname,p_key1,p_key2)

	return HttpResponse("<script>alert('Data Uploaded Successfully');window.location.href='/show_home_user/'</script>")

model = joblib.load("enc_app/static/assets/CNN_Sync_users_5_relu_Adam_FL_Model.h5")


def decryption(name,obtain_key,loc_):
	loc_path=loc+name
	g_folder=os.listdir(loc_path)
	g_folder=g_folder[0]
	f_path=loc_path+"/"+g_folder
	##print("f decrypt path : ",f_path)
	decrypt_file(f_path,obtain_key,name,g_folder)


@never_cache
def view_data(request):
	if 'uid' in request.session:
		username=request.session["uid"]
		obj=Hashes.objects.filter(uname=username)
		return render(request,'view_data.html',{'obj':obj}) 
	else:
		return render(request,'login.html')


def get_from_ipfs(hash_):
	api = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
	api.get(hash_)
	with open(hash_, 'rb') as fh:
		response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
		return hash_

def get_false_data(uname):
	import shutil   

	source_folder = 'enc_app/static/Datas/'+uname
	destination_folder = 'Downloads/'+uname
	# Check if the destination folder already exists
	if os.path.exists(destination_folder):
		shutil.rmtree(destination_folder)  # Delete the existing folder and its contents

	shutil.copytree(source_folder, destination_folder)


def download(request):
	h_id=request.POST.get("h_id")
	hash_=request.POST.get("hash_val")
	uname=request.POST.get("uname")
	p_key=request.POST.get("p_key")

	res=get_from_ipfs(hash_)
	os.remove(res)
	try:
		decryption(uname,p_key,hash_)
	except:
		get_false_data(uname)
	
	return HttpResponse("<script>alert('Downloaded Successfully! Check Downloads Folder');window.location.href='/view_data/'</script>")


@never_cache
def display_ep_page(request):
	if 'uid' in request.session:

		return render(request,'prediction.html') 
	else:
		return render(request,'login.html')


# Function to preprocess user input data
def preprocess_input(input_data):
  
	input_list = input_data.split(',')
	
	# Convert the list into a DataFrame with appropriate column names
	input_df = pd.DataFrame([input_list], columns=['ID', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
												   'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
												   'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
												   'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS'])
	
	return input_df

# Function to make predictions on a single row of input data
def predict_single_row(input_data):
	# Preprocess the input data
	preprocessed_input = preprocess_input(input_data)
	
	# Make predictions using the loaded model
	predictions = model.predict(preprocessed_input)
	
	# Return the predictions
	return predictions

def perform_prediction(request):
	feat=request.POST.get("features")

	# Make predictions on the user input data
	prediction = predict_single_row(feat)

	# Display the prediction

	prediction=prediction[0]
	print("Prediction:", prediction)

	if prediction==0:
		print("Eligible")
		return HttpResponse("<script>alert('Eligible');window.location.href='/display_ep_page/'</script>")
	else:
		print("Not Eligible")
		return HttpResponse("<script>alert('Not Eligible');window.location.href='/display_ep_page/'</script>")


def fetch_data(request):
	try:
		c_id=request.POST.get("c_id")
		uname=request.session["uid"]
		destination_folder = 'Downloads/'+uname
		c_id=int(c_id)
		files = os.listdir(destination_folder)

		# Find the CSV file in the folder
		csv_files = [file for file in files if file.endswith('.csv')]

		if len(csv_files) == 0:
			print("No CSV file found in the folder.")
		else:
			csv_filename = csv_files[0]
			csv_filepath = os.path.join(destination_folder, csv_filename)

			df=pd.read_csv(csv_filepath)
			print(df)
			print("*******")
			print(c_id)
			matching_row = df[df['ID'] == c_id]
			column_names = matching_row.columns.tolist()
			values = matching_row.values.tolist()[0] if not matching_row.empty else []

			excluded_columns = ['STATUS', 'ID']
			filtered_column_value_pairs = [(col, val) for col, val in zip(column_names, values) if col not in excluded_columns]

			context = {
			    'column_value_pairs': filtered_column_value_pairs
			}
			if not context['column_value_pairs']:
				return HttpResponse("<script>alert('Provide Correct Customer ID');window.location.href='/view_data/'</script>")
			
			else:
				print("context :::",context)
				return render(request, 'view_results.html', context)
	except:
		return HttpResponse("<script>alert('Perform Decryption First');window.location.href='/view_data/'</script>")

