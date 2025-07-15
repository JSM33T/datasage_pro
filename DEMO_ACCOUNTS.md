# Authentication System with Demo Accounts

## Overview
The authentication system now supports both LDAP authentication and hardcoded demo accounts for demonstration purposes.

## Demo Accounts

### SuperAdmin Account
- **Username:** `superadmin`
- **Password:** `superadmin123`
- **Role:** `superadmin`
- **Access:** Full access to all features including document management

### Regular User Account
- **Username:** `demouser`
- **Password:** `user123`
- **Role:** `user`
- **Access:** Can use chat features, view and download documents, but cannot upload/delete documents

## Authentication Flow

1. **Demo Account Check First**: The system first checks if the provided credentials match any hardcoded demo accounts
2. **LDAP Fallback**: If no demo account matches, it attempts LDAP authentication
3. **JWT Token Generation**: Successfully authenticated users receive a JWT token with their role and details

## Access Control

### SuperAdmin (`superadmin`)
- ✅ Document upload, delete, reindex
- ✅ All indexing operations
- ✅ Chat operations (user-specific sessions)
- ✅ Document viewing, searching, downloading
- ✅ Full system access

### Admin (LDAP users with admin groups)
- ✅ Document upload, delete, reindex
- ✅ All indexing operations
- ✅ Chat operations (user-specific sessions)
- ✅ Document viewing, searching, downloading

### User (`demouser` and regular LDAP users)
- ❌ Document upload, delete, reindex
- ❌ Indexing operations
- ✅ Chat operations (user-specific sessions)
- ✅ Document viewing, searching, downloading

## Frontend Changes

### Navigation
- Document management link now shows for both `admin` and `superadmin` roles
- Regular users only see chat and home navigation

### Login Screen
- Added demo account credentials display
- Shows both SuperAdmin and User demo accounts for easy testing

## API Endpoints Access

### SuperAdmin Only
- All endpoints (no restrictions)

### Admin Required
- `POST /api/document/add` - Upload documents
- `POST /api/document/delete` - Delete documents
- `POST /api/document/reindex` - Reindex documents
- `POST /api/indexing/*` - All indexing operations

### User Access (All Authenticated Users)
- `GET /api/document/list` - View documents
- `GET /api/document/search` - Search documents
- `GET /api/document/download` - Download documents
- `GET /api/document/by_ids` - Get documents by IDs
- `GET /api/chat_session/*` - Chat session operations
- `POST /api/chat_session/*` - Chat session operations
- `GET /api/chat/*` - Chat operations
- `POST /api/chat/*` - Chat operations

## Implementation Details

### Backend (main.py)
```python
# Demo account authentication
if username == "superadmin" and password == "superadmin123":
    # SuperAdmin account
    role = "superadmin"
elif username == "demouser" and password == "user123":
    # Regular user account
    role = "user"
else:
    # LDAP authentication
    role = "admin" if "admin" in user_groups else "user"
```

### Middleware Access Control
```python
# SuperAdmin has full access
if user_payload.get('username') == 'superadmin':
    pass  # No restrictions

# Admin operations require admin role or superadmin
elif path.startswith("/api/document/add"):
    if not (is_admin or is_superadmin):
        return 403
```

## Testing

### SuperAdmin Test
1. Login with `superadmin` / `superadmin123`
2. Verify access to Document management
3. Test document upload/delete functionality
4. Verify chat operations work with user-specific sessions

### User Test
1. Login with `demouser` / `user123`
2. Verify NO access to Document management
3. Verify can view/search/download documents
4. Verify chat operations work with user-specific sessions

### LDAP Test
1. Login with LDAP credentials
2. Role determined by LDAP groups
3. Same access control as respective roles

## Security Notes

- Demo accounts are hardcoded for demonstration only
- JWT tokens include user role and details
- User-specific sessions prevent cross-user data access
- Proper role-based access control implemented
- LDAP authentication maintains same security as before
