from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode
# import hashlib
import json

# SALT="THANKMRGOOSE"
SECRET_KEY=b'\xcd4H\xe6\xff\xd5\x18ff\xb4\xf0\xdb\x9d\x92ux\x8b\xfa\x11\xcb0\xa1g\xa8M}\x87\x9c[\x0c\x85\x13'
# print(SECRET_KEY)

def encrypt_chacha(text):
    cipher = ChaCha20.new(key=SECRET_KEY)
    ciphertext = cipher.encrypt(text.encode())
    nonce = b64encode(cipher.nonce).decode('utf-8')
    ct = b64encode(ciphertext).decode('utf-8')
    result = json.dumps({'nonce': nonce, 'ciphertext': ct})
    return result

def decrypt_chacha(encrypted_data):
    try:
        b64 = json.loads(encrypted_data)
        nonce = b64decode(b64['nonce'])
        ciphertext = b64decode(b64['ciphertext'])
        cipher = ChaCha20.new(key=SECRET_KEY, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext)
        return plaintext
    except:
        print("Incorrect decryption")

# def get_private_key(secret_key, salt):
#     return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, 32)

# # AES encrypt plaintext
# def encrypt_with_AES(message, secret_key=SECRET_KEY, salt=SALT):
#     private_key = get_private_key(secret_key, salt)
#     message = pad(message)
#     cipher = AES.new(private_key, AES.MODE_CBC, iv)
#     cipher_bytes = base64.b64encode(iv + cipher.encrypt(message))
#     return bytes.decode(cipher_bytes)

# # decrypt AES encrypted data to plaintext
# def decrypt_with_AES(encoded, secret_key=SECRET_KEY, salt=SALT):
#     private_key = get_private_key(secret_key, salt)
#     cipher_text = base64.b64decode(encoded)
#     iv = cipher_text[:AES.block_size]
#     cipher = AES.new(private_key, AES.MODE_CBC, iv)
#     plain_bytes = unpad(cipher.decrypt(cipher_text[block_size:]));
#     return bytes.decode(plain_bytes)

# test = "this is working"
# res = encrypt(test)
# print(res)
# print(decrypt(res))
