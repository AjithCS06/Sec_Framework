import ipfshttpclient
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import sqlite3
from c import *

db = sqlite3.connect('db.sqlite3')
print("Successfully Connected")

def insert_db(username,hash_value):
	cursor=db.cursor()
	sql="""insert into enc_app_hashes(uname,hash_val)values('%s','%s')"""%(username,hash_value)
	#print (sql)
	try:
		cursor.execute(sql)
		db.commit()
		print ("Inserted.")
	except Exception as e:
		db.rollback()
		print ("error",e)



def encryption(input_file,i_path,path,key,uname,folder_path):
	# Create AES cipher object
	# convert string to byte
	#print("enc key:",key)
	bytes_key = key.encode('utf-8')

	cipher = AES.new(bytes_key, AES.MODE_ECB)

	# Read the input file
	with open(i_path+input_file, 'rb') as file:
		plaintext = file.read()

	# Pad the plaintext
	padded_plaintext = pad(plaintext, AES.block_size)

	# Encrypt the padded plaintext
	ciphertext = cipher.encrypt(padded_plaintext)

	output_file=path+"encrypted_"+input_file
	# Save the encrypted file
	with open(output_file, 'wb') as file:
		file.write(ciphertext)

	print(output_file)

	api = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
	new_file = api.add(output_file)
	print(new_file)
	hash1=new_file.get('Hash')

	print("hash : ",hash1)
	insert_db(uname,hash1)

	print('Encrypted successfully.')



def perform_encryption():
	path='enc_app/static/enc_params/'
	os.makedirs(path, exist_ok=True)
	o_path='enc_app/static/Datas/'
	parms_path='enc_app/static/params/'
	get_list=os.listdir(parms_path)
	print(get_list)
	key_path='enc_app/static/User_Keys/'
	for i in get_list:
		get_name=i.split('.')[0]
		get_name_ = get_name.replace("_", "")
		#print(get_name_)
		get_file=os.listdir(key_path+get_name_)
		for j in get_file:
			#print(j)
			file_path=key_path+get_name_+"/"+j
			with open(file_path, 'r') as file:
				obtain_k = file.readline().strip()
			get_key=obtain_k[:16]
			bytes_key = get_key.encode('utf-8')
			o_path_=o_path+get_name_
			g_folder=os.listdir(o_path_)
			g_folder=g_folder[0]
			f_path=o_path_+"/"+g_folder
			print("f encrypt path :",f_path)
			encrypt_file(f_path,obtain_k)
			encryption(i,parms_path,path,get_key,get_name_,o_path)




# def encrypt_folder(folder_path, key):

# 	# path='enc_app/static/E_Datas/'
# 	# os.makedirs(path, exist_ok=True)

# 	# Create a cipher object with AES algorithm and CBC mode
# 	cipher = AES.new(key, AES.MODE_CBC)

# 	for root, dirs, files in os.walk(folder_path):
# 		# print("&&&&&&&&&&")
# 		# print(root)
# 		# print(dirs)
# 		# print(files)
# 		for file in files:
# 			file_path = os.path.join(root, file)

# 			# Read the file content
# 			with open(file_path, 'rb') as f:
# 				plaintext = f.read()

# 			# Encrypt the content
# 			ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

# 			# Write the encrypted content back to the file
# 			with open(file_path, 'wb') as f:
# 				f.write(ciphertext)
