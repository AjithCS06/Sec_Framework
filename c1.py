

import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import shutil



def decrypt_file(file_path, obtain_key,name,file_name):
	key = obtain_key[:16]
	bytes_key = key.encode('utf-8')
	print("bytes key: ", bytes_key)

	download_path="Downloads/"+name
	if os.path.exists(download_path):
		shutil.rmtree(download_path)  # Delete the existing folder and its contents
	
	os.makedirs(download_path, exist_ok=True)

	# Read the file content
	with open(file_path, 'rb') as f:
		data = f.read()

	# Extract the IV and ciphertext
	iv = data[:AES.block_size]
	ciphertext = data[AES.block_size:]

	# Create a cipher object with AES algorithm, CBC mode, and the extracted IV
	cipher = AES.new(bytes_key, AES.MODE_CBC, iv)

	# Decrypt the content
	plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size, style='pkcs7')

	# Write the decrypted content back to the file
	with open(download_path+"/"+file_name, 'wb') as f:
		f.write(plaintext)


# def decrypt_folder(folder_path, obtain_key,name):
# 	print("folderpath :",folder_path)
# 	key=obtain_key[:16]
# 	bytes_key = key.encode('utf-8')
# 	print("bytes key :",bytes_key)
# 	download_path="Downloads/"+name
# 	if os.path.exists(download_path):
# 	    shutil.rmtree(download_path)  # Delete the existing folder and its contents
	
# 	os.makedirs(download_path, exist_ok=True)
# 	for root, dirs, files in os.walk(folder_path):
# 		for file in files:
# 			file_path = os.path.join(root, file)
# 			#print(file_path)

# 			# Read the file content
# 			with open(file_path, 'rb') as f:
# 				data = f.read()
# 				#print(data)
# 			#print(AES.block_size)
# 			# Extract the IV and ciphertext
# 			iv = data[:AES.block_size]
# 			#print(iv)
# 			ciphertext = data[AES.block_size:]

# 			# Create a cipher object with AES algorithm, CBC mode, and the extracted IV
# 			cipher = AES.new(bytes_key, AES.MODE_CBC, iv)

# 			# Decrypt the content
# 			plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size, style='pkcs7')


# 			# Write the decrypted content back to the file
# 			with open(download_path+"/"+file, 'wb') as f:
# 				f.write(plaintext)
	
# 			data=''




#decrypt_folder('1','0x9de435cc0fae4626dc23d2a60f211382cad8665a61a56492f196cac9653f547f1')