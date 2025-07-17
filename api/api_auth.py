import os
import requests
import json
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import logging
import uuid
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIAuthConfig:
    def __init__(self):
        self.jwt_token_url = "https://revalposapi.revalerp.com/api/JWTToken"
        self.login_url = "https://revalposapi.revalerp.com/api/Login"
        self.site_url = "https://revalsys.revalerp.com"
        self.country_code = "ind"
        self.currency_code = "inr"
        self.language_code = "eng"
        self.jwt_secret = os.getenv("JWT_SECRET", "RevalSysSecretKey")
        self.jwt_expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", "48"))

class APIAuthenticator:
    def __init__(self):
        self.config = APIAuthConfig()
        self.session = requests.Session()
        
        # Common headers for all requests
        self.common_headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9,en-IN;q=0.8,tr;q=0.7',
            'content-type': 'application/json',
            'origin': 'https://revalsys.revalerp.com',
            'priority': 'u=1, i',
            'referer': 'https://revalsys.revalerp.com/',
            'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0'
        }
    
    def get_jwt_token(self) -> Optional[str]:
        """
        First request: Get JWT token for authentication
        """
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            
            payload = {
                "SessionID": session_id,
                "SiteURL": self.config.site_url,
                "CountryCode": self.config.country_code,
                "CurrencyCode": self.config.currency_code,
                "LanguageCode": self.config.language_code
            }
            
            logger.info(f"Step 1: Requesting JWT token with session ID: {session_id}")
            
            response = self.session.post(
                self.config.jwt_token_url,
                headers=self.common_headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Step 1: Response status: {response.status_code}")
            logger.info(f"Step 1: Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            logger.info(f"Step 1: Response data: {response_data}")
            
            # Check if response contains token
            if 'SecurityToken' in response_data:
                jwt_token = response_data['SecurityToken']
                logger.info("Step 1: JWT token obtained successfully from SecurityToken")
                return jwt_token
            elif 'JwtToken' in response_data:
                jwt_token = response_data['JwtToken']
                logger.info("Step 1: JWT token obtained successfully from JwtToken")
                return jwt_token
            elif 'Data' in response_data and len(response_data['Data']) > 0:
                jwt_token = response_data['Data'][0].get('JwtToken') or response_data['Data'][0].get('SecurityToken')
                if jwt_token:
                    logger.info("Step 1: JWT token obtained successfully from Data array")
                    return jwt_token
            
            logger.error(f"Step 1: JWT token not found in response: {response_data}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Step 1: Request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Step 1: JSON decode error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Step 1: Unexpected error: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Second request: Authenticate user with username and password
        """
        try:
            # Step 1: Get JWT token
            jwt_token = self.get_jwt_token()
            if not jwt_token:
                raise Exception("Failed to obtain JWT token")
            
            # Step 2: Authenticate user
            logger.info(f"Step 2: Authenticating user: {username}")
            
            # Encrypt password using AES encryption
            encrypted_password = self.encrypt_password(password)
            logger.info(f"Step 2: Password encrypted for user: {username}")
            
            # Add authorization header
            auth_headers = self.common_headers.copy()
            auth_headers['authorization'] = f'Bearer {jwt_token}'
            auth_headers['cache-control'] = 'no-cache'
            auth_headers['contenttype'] = 'application/json'
            auth_headers['expires'] = 'Sat, 01 Jan 2000 00:00:00 GMT'
            auth_headers['pragma'] = 'no-cache'
            
            login_payload = {
                "UserName": username,
                "UserPassword": encrypted_password,  # Use encrypted password
                "CountryCode": self.config.country_code,
                "CurrencyCode": self.config.currency_code,
                "LanguageCode": self.config.language_code
            }
            
            response = self.session.post(
                self.config.login_url,
                headers=auth_headers,
                json=login_payload,
                timeout=30
            )
            
            logger.info(f"Step 2: Response status: {response.status_code}")
            logger.info(f"Step 2: Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            logger.info(f"Step 2: Login response: {response_data}")
            
            # Check if authentication was successful
            if (response_data.get('ReturnCode') == 0 and 
                response_data.get('ReturnMessage') == 'success' and 
                response_data.get('Data') and 
                len(response_data['Data']) > 0):
                
                user_data = response_data['Data'][0]
                
                # Extract user information
                user_details = {
                    'username': username,
                    'displayName': f"{user_data.get('FirstName', '')} {user_data.get('LastName', '')}".strip(),
                    'firstName': user_data.get('FirstName', ''),
                    'lastName': user_data.get('LastName', ''),
                    'email': username,  # Assuming username is email
                    'department': 'Not specified',
                    'groups': ['user'],
                    'api_token': user_data.get('JwtToken')  # Store the API token
                }
                
                logger.info(f"Step 2: Authentication successful for {username}")
                return user_details
            else:
                error_msg = response_data.get('ReturnMessage', 'Authentication failed')
                logger.error(f"Step 2: Authentication failed: {error_msg}")
                raise Exception(f"Authentication failed: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Step 2: Request failed: {str(e)}")
            raise Exception(f"Authentication request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Step 2: JSON decode error: {str(e)}")
            raise Exception(f"Invalid response format: {str(e)}")
        except Exception as e:
            logger.error(f"Step 2: Authentication error: {str(e)}")
            raise e
    
    def generate_jwt_token(self, user_details: Dict[str, Any]) -> str:
        """
        Generate JWT token for authenticated user
        """
        try:
            payload = {
                'username': user_details['username'],
                'displayName': user_details['displayName'],
                'firstName': user_details['firstName'],
                'lastName': user_details['lastName'],
                'email': user_details['email'],
                'department': user_details['department'],
                'api_token': user_details.get('api_token'),  # Store API token in JWT
                'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
            return token
            
        except Exception as error:
            logger.error(f"JWT token generation error: {str(error)}")
            raise error
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        """
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception('Token has expired')
        except jwt.InvalidTokenError:
            raise Exception('Invalid token')
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connectivity
        """
        try:
            logger.info("Testing connection to API endpoints")
            
            # Test JWT token endpoint
            jwt_token = self.get_jwt_token()
            
            if jwt_token:
                return {
                    'success': True,
                    'message': 'Successfully connected to API endpoints',
                    'jwt_token_url': self.config.jwt_token_url,
                    'login_url': self.config.login_url
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to obtain JWT token',
                    'jwt_token_url': self.config.jwt_token_url
                }
                
        except Exception as error:
            logger.error(f"Connection test failed: {str(error)}")
            return {
                'success': False,
                'message': f'Connection failed: {str(error)}',
                'jwt_token_url': self.config.jwt_token_url
            }
    
    def encrypt_password(self, password: str, key: str = "RevalKey") -> str:
        """
        Encrypt password using AES encryption with the given key (similar to CryptoJS.AES.encrypt)
        This implementation tries to match CryptoJS.AES.encrypt behavior
        """
        try:
            # CryptoJS uses UTF-8 encoding and PBKDF2 for key derivation
            # For simplicity, we'll use a direct approach that should work with the API
            
            # Generate salt (8 random bytes)
            salt = os.urandom(8)
            
            # Derive key and IV using a method similar to CryptoJS
            key_iv = self._derive_key_iv(key.encode('utf-8'), salt)
            derived_key = key_iv[:32]  # First 32 bytes for key
            iv = key_iv[32:48]  # Next 16 bytes for IV
            
            # Convert password to bytes
            password_bytes = password.encode('utf-8')
            
            # Create cipher
            cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad the password to be multiple of 16 bytes (AES block size)
            padder = padding.PKCS7(128).padder()
            padded_password = padder.update(password_bytes)
            padded_password += padder.finalize()
            
            # Encrypt the password
            encrypted_password = encryptor.update(padded_password) + encryptor.finalize()
            
            # Create the format similar to CryptoJS: "Salted__" + salt + encrypted_data
            salted_prefix = b'Salted__'
            final_data = salted_prefix + salt + encrypted_password
            
            # Convert to base64 string (similar to CryptoJS output)
            encrypted_base64 = base64.b64encode(final_data).decode('utf-8')
            
            logger.info(f"Password encrypted successfully")
            return encrypted_base64
            
        except Exception as e:
            logger.error(f"Password encryption failed: {str(e)}")
            raise Exception(f"Password encryption failed: {str(e)}")
    
    def _derive_key_iv(self, password: bytes, salt: bytes) -> bytes:
        """
        Derive key and IV using MD5 (similar to CryptoJS default)
        """
        key_iv = b''
        prev = b''
        
        while len(key_iv) < 48:  # 32 bytes for key + 16 bytes for IV
            prev = hashlib.md5(prev + password + salt).digest()
            key_iv += prev
            
        return key_iv
    
# Global API authenticator instance
api_authenticator = APIAuthenticator()
