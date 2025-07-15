#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.api_auth import APIAuthenticator

def test_password_encryption():
    """Test password encryption"""
    print("Testing password encryption...")
    
    authenticator = APIAuthenticator()
    
    # Test password encryption
    test_password = "testpassword123"
    encrypted = authenticator.encrypt_password(test_password)
    
    print(f"Original password: {test_password}")
    print(f"Encrypted password: {encrypted}")
    print(f"Encrypted length: {len(encrypted)}")
    
    # Test if it starts with expected base64 format
    import base64
    try:
        decoded = base64.b64decode(encrypted)
        print(f"Base64 decoded length: {len(decoded)}")
        print(f"Starts with 'Salted__': {decoded.startswith(b'Salted__')}")
    except Exception as e:
        print(f"Base64 decode error: {e}")

def test_api_connection():
    """Test API connection"""
    print("\nTesting API connection...")
    
    authenticator = APIAuthenticator()
    result = authenticator.test_connection()
    
    print(f"Connection test result: {result}")

if __name__ == "__main__":
    test_password_encryption()
    test_api_connection()
