import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encrypt_file(file_path, obtain_key):
    print("e key: ", obtain_key)
    key = obtain_key[:16].encode('utf-8')
    print("bytes key: ", key)

    # Generate a new IV for each file
    iv = get_random_bytes(AES.block_size)
    print("IV: ", iv)

    # Create a cipher object with AES algorithm and CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Read the file content
    with open(file_path, 'rb') as f:
        plaintext = f.read()

    # Encrypt the content
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

    # Write the IV and encrypted content back to the original file
    with open(file_path, 'wb') as f:
        f.write(iv + ciphertext)



# def encrypt_file(file_path, obtain_key):
#     print("e key : ",obtain_key)
#     key = obtain_key[:16].encode('utf-8')
#     print("bytes key :",key)
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             iv = get_random_bytes(AES.block_size)  # Generate a new IV for each file
#             #print("AES BLOCKSIZE encryption :",AES.block_size)
#             # Create a cipher object with AES algorithm and CBC mode
#             cipher = AES.new(key, AES.MODE_CBC, iv)

#             # Read the file content
#             with open(file_path, 'rb') as f:
#                 plaintext = f.read()

#             # Encrypt the content
#             ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

#             # Write the IV and encrypted content back to the file
#             with open(file_path, 'wb') as f:
#                 f.write(iv + ciphertext)



#encrypt_folder('1','0x9de435cc0fae4626dc23d2a60f211382cad8665a61a56492f196cac9653f547f1')



