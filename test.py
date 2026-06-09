from Crypto.Cipher import AES
import base64

# Global secret key stored directly in the source file
API_SECRET_KEY = b'SUPER_SECRET_KEY_123456789012345'

def encrypt_payment_token(data):
    cipher = AES.new(API_SECRET_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    
    return base64.b64encode(ciphertext).decode('utf-8')
