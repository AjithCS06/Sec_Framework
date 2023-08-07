import ipfshttpclient
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import os
loc="enc_app/static/Datas/"
def decryption(name,obtain_key):
	loc_path=loc+name
	print("lo path : ",loc_path)
	key=obtain_key[:16]
	print("dec_key :",key)
	bytes_key = key.encode('utf-8')
	print("dec_byte_key:",bytes_key)
	download_path="Downloads/"+name
	os.makedirs(download_path, exist_ok=True)

	image=loc_path+"/"+str(0)+"/img_1.jpg"
	with open(image, 'rb') as f:
		data = f.read()

	iv = data[:AES.block_size]
	print("******")
	print(iv)
	print(AES.block_size)
	ciphertext = data[AES.block_size:]

	cipher = AES.new(bytes_key, AES.MODE_CBC, iv)

	# Decrypt the content
	plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

	# Write the decrypted content back to the file
	with open("deccc.jpg", 'wb') as f:
		f.write(plaintext)

decryption("user1","0x9de435cc0fae4626dc23d2a60f211382cad8665a61a56492f196cac9653f547f1")

	# for root, dirs, files in os.walk(loc_path):
	# 	print(root)
	# 	print(dirs)
	# 	for file in files:
	# 		print(file)
	# 		file_path = os.path.join(root, file)
	# 		final_path=os.path.join(download_path,file)
	# 		print("file_path",file_path)
	# 		print("final_path",final_path)
	# 		data=''
	# 		# Read the file content
	# 		with open(file_path, 'rb') as f:
	# 			data = f.read()

	# 		iv = data[:AES.block_size]
	# 		print("******")
	# 		print(iv)
	# 		print(AES.block_size)
	# 		ciphertext = data[AES.block_size:]

	# 		cipher = AES.new(bytes_key, AES.MODE_CBC, iv)

	# 		# Decrypt the content
	# 		plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

	# 		# Write the decrypted content back to the file
	# 		with open(final_path, 'wb') as f:
	# 			f.write(plaintext)