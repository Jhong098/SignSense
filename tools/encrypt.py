from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode
import json

SECRET_KEY=b'\xcd4H\xe6\xff\xd5\x18ff\xb4\xf0\xdb\x9d\x92ux\x8b\xfa\x11\xcb0\xa1g\xa8M}\x87\x9c[\x0c\x85\x13'

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

