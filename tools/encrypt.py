from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

SECRET_KEY = '88ab02664ae3197ec531da8bd7ea0b5a'.encode()

class DecryptionError(Exception):
    pass

# takes a string and encrypts into a bytearray
def encrypt_chacha(text):
    nonce_rfc7539 = get_random_bytes(12)
    cipher = ChaCha20.new(key=SECRET_KEY, nonce=nonce_rfc7539)
    ciphertext = cipher.encrypt(text.encode())
    nonce = cipher.nonce
    ct = ciphertext
    result = bytearray(nonce + ct)
    return result

# takes bytearray and decrypts into string
def decrypt_chacha(encrypted_data):
    try:
        b64 = bytearray(encrypted_data)
        nonce = b64[:12]
        ciphertext = b64[12:]
        cipher = ChaCha20.new(key=SECRET_KEY, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext).decode()
        return plaintext
    except:
        raise DecryptionError

def test():
    text = "TESTING"
    encrypted = encrypt_chacha(text)
    print(encrypted)
    decrypted = decrypt_chacha(encrypted)
    print(decrypted)
    assert(text == decrypted)

# test()
