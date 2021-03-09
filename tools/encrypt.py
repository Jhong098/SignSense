from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode
import json

SECRET_KEY=b'\xcd4H\xe6\xff\xd5\x18ff\xb4\xf0\xdb\x9d\x92ux\x8b\xfa\x11\xcb0\xa1g\xa8M}\x87\x9c[\x0c\x85\x13'

# takes a string and encrypts into bytes
def encrypt_chacha(text):
    nonce_rfc7539 = get_random_bytes(12)
    cipher = ChaCha20.new(key=SECRET_KEY, nonce=nonce_rfc7539)
    ciphertext = cipher.encrypt(text.encode())
    nonce = b64encode(cipher.nonce)
    ct = b64encode(ciphertext)
    result = nonce + ct
    return result

# takes bytes and decrypts into string
def decrypt_chacha(encrypted_data):
    try:
        b64 = b64decode(encrypted_data)
        nonce = b64[:12]
        ciphertext = b64[12:]
        cipher = ChaCha20.new(key=SECRET_KEY, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext).decode()
        return plaintext
    except:
        print("Incorrect decryption")

def test():
    text = "TESTING"
    encrypted = encrypt_chacha(text)
    print(encrypted)
    decrypted = decrypt_chacha(encrypted)
    print(decrypted)
    assert(text == decrypted)

