import os
import ldap3
from ldap3 import Server, Connection, ALL, SUBTREE
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LDAPConfig:
    def __init__(self):
        self.url = os.getenv("LDAP_URL", "ldap://DEVTEAM.IDA:389")
        self.admin_dn = os.getenv("LDAP_ADMIN_DN", "CN=suresh,DC=DEVTEAM,DC=IDA")
        self.admin_password = os.getenv("LDAP_ADMIN_PASSWORD", "Chintal@92325")
        self.base_dn = os.getenv("LDAP_BASE_DN", "DC=DEVTEAM,DC=IDA")
        self.user_search_base = os.getenv("LDAP_USER_SEARCH_BASE", "DC=DEVTEAM,DC=IDA")
        self.domain = os.getenv("LDAP_DOMAIN", "DEVTEAM.IDA")
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

class LDAPAuthenticator:
    def __init__(self):
        self.config = LDAPConfig()
        self.directory_searcher = None
    
    def get_directory_searcher(self, username: str, password: str, domain: str) -> Optional[Connection]:
        """
        Create LDAP connection (equivalent to DirectorySearcher in C#)
        """
        try:
            logger.info(f"Step 3: GetDirectorySearcher Method inside if - Domain: {domain}, Username: {username}")
            
            if not self.directory_searcher:
                ldap_url = f"ldap://{domain}"
                logger.info(f"Step 4: Creating LDAP connection to {ldap_url}")
                
                server = Server(ldap_url, get_info=ALL)
                user_dn = f"{username}@{domain}"
                
                self.directory_searcher = Connection(
                    server,
                    user=user_dn,
                    password=password,
                    auto_bind=True,
                    authentication='SIMPLE'
                )
                
                logger.info('Step 4: DirectorySearcher created successfully')
                return self.directory_searcher
            else:
                logger.info('Step 5: DirectorySearcher already exists, returning existing instance')
                return self.directory_searcher
                
        except Exception as ex:
            logger.error(f"Step 6: GetDirectorySearcher Method catch - Error: {str(ex)}")
            return None
    
    def authenticate_with_directory_searcher(self, username: str, password: str, domain: str = None) -> Dict[str, Any]:
        """
        Enhanced LDAP Authentication with DirectorySearcher pattern
        """
        try:
            target_domain = domain or self.config.domain
            
            # Step 1: Get DirectorySearcher instance
            searcher = self.get_directory_searcher(username, password, target_domain)
            if not searcher:
                raise Exception('Failed to create DirectorySearcher')
            
            logger.info(f"Authentication successful for {username}@{target_domain}")
            
            # Step 2: Search for user details
            user_details = self.search_user_details(searcher, username, target_domain)
            
            return user_details
            
        except Exception as error:
            logger.error(f"DirectorySearcher authentication error: {str(error)}")
            raise error
        finally:
            # Clean up DirectorySearcher instance
            if self.directory_searcher:
                try:
                    self.directory_searcher.unbind()
                    self.directory_searcher = None
                except Exception as cleanup_error:
                    logger.error(f"Cleanup error: {str(cleanup_error)}")
    
    def search_user_details(self, searcher: Connection, username: str, domain: str) -> Dict[str, Any]:
        """
        Search for user details using DirectorySearcher pattern
        """
        try:
            search_filter = f"(|(sAMAccountName={username})(userPrincipalName={username}@{domain}))"
            attributes = ['sAMAccountName', 'cn', 'givenName', 'mail', 'department', 'memberOf', 'displayName']
            
            logger.info(f"Searching for user with filter: {search_filter}")
            
            searcher.search(
                search_base=self.config.base_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=attributes
            )
            
            if searcher.entries:
                entry = searcher.entries[0]
                
                # Extract user details
                user_details = {
                    'username': str(entry.sAMAccountName) if entry.sAMAccountName else username,
                    'displayName': str(entry.displayName) if entry.displayName else str(entry.cn) if entry.cn else str(entry.givenName) if entry.givenName else username,
                    'email': str(entry.mail) if entry.mail else f"{username}@{domain}",
                    'department': str(entry.department) if entry.department else 'Not specified',
                    'groups': [str(group) for group in entry.memberOf] if entry.memberOf else []
                }
                
                logger.info(f"User found: {user_details}")
                return user_details
            else:
                raise Exception('User not found in directory')
                
        except Exception as error:
            logger.error(f"Search user details error: {str(error)}")
            raise error
    
    def generate_jwt_token(self, user_details: Dict[str, Any], domain: str) -> str:
        """
        Generate JWT token for authenticated user
        """
        try:
            payload = {
                'username': user_details['username'],
                'displayName': user_details['displayName'],
                'email': user_details['email'],
                'department': user_details['department'],
                'domain': domain,
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
        Test LDAP connectivity
        """
        try:
            logger.info(f"Testing connection to LDAP domain: {self.config.domain}")
            
            server = Server(self.config.url, get_info=ALL)
            test_conn = Connection(
                server,
                user=self.config.admin_dn,
                password=self.config.admin_password,
                auto_bind=True,
                authentication='SIMPLE'
            )
            
            test_conn.unbind()
            
            return {
                'success': True,
                'message': f'Successfully connected to {self.config.domain}',
                'domain': self.config.domain,
                'url': self.config.url
            }
            
        except Exception as error:
            logger.error(f"Connection test failed: {str(error)}")
            return {
                'success': False,
                'message': f'Connection failed: {str(error)}',
                'domain': self.config.domain
            }

# Global LDAP authenticator instance
ldap_authenticator = LDAPAuthenticator()
