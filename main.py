import os
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime

from api import chat_session, document, chat, indexing
from api.ldap_auth import ldap_authenticator

app = FastAPI()

# ===== Static folders =====
app.mount("/static", StaticFiles(directory="static/browser"), name="static")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# ===== Root index.html =====
@app.get("/")
def serve_root():
    return FileResponse(Path("static/browser/index.html"))

# ===== LDAP Login endpoint =====
@app.post("/api/auth/login")
async def login(payload: dict = Body(...)):
    username = payload.get("username")
    password = payload.get("password")
    domain = payload.get("domain")  # Optional, will use default if not provided
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    try:
        print(f"Authentication request for user: {username}")
        
        # Check for hardcoded demo accounts first
        if username == "superadmin" and password == "superadmin123":
            # Hardcoded superadmin account
            user_details = {
                'username': 'superadmin',
                'displayName': 'Super Administrator',
                'email': 'superadmin@demo.com',
                'department': 'IT Administration',
                'groups': ['superadmin']
            }
            token = ldap_authenticator.generate_jwt_token(user_details, "demo.local")
            role = "superadmin"
            
            print(f"Hardcoded superadmin authentication successful")
            
            return {
                "success": True,
                "message": "Authentication successful for demo superadmin",
                "user": user_details,
                "token": token,
                "role": role
            }
        
        elif username == "demouser" and password == "user123":
            # Hardcoded regular user account
            user_details = {
                'username': 'demouser',
                'displayName': 'Demo User',
                'email': 'demouser@demo.com',
                'department': 'General',
                'groups': ['user']
            }
            token = ldap_authenticator.generate_jwt_token(user_details, "demo.local")
            role = "user"
            
            print(f"Hardcoded demo user authentication successful")
            
            return {
                "success": True,
                "message": "Authentication successful for demo user",
                "user": user_details,
                "token": token,
                "role": role
            }
        
        else:
            # Try LDAP authentication for other users
            user_details = ldap_authenticator.authenticate_with_directory_searcher(
                username=username, 
                password=password, 
                domain=domain
            )
            
            # Generate JWT token
            token = ldap_authenticator.generate_jwt_token(user_details, domain or ldap_authenticator.config.domain)
            
            # Determine role based on user groups or other criteria
            role = "admin" if "admin" in user_details.get('groups', []) else "user"
            
            print(f"LDAP authentication successful for {username}@{domain or ldap_authenticator.config.domain}")
            
            return {
                "success": True,
                "message": f"Authentication successful for {domain or ldap_authenticator.config.domain}",
                "user": user_details,
                "token": token,
                "role": role
            }
        
    except Exception as error:
        print(f"Authentication failed for {username}: {str(error)}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(error)}")


# ===== Test LDAP connection endpoint =====
@app.get("/api/auth/test-connection")
async def test_ldap_connection():
    try:
        result = ldap_authenticator.test_connection()
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(error)}")


# ===== Health check endpoint =====
@app.get("/api/health")
async def health_check():
    return {
        "status": "OK",
        "service": "LDAP Authentication API",
        "domain": ldap_authenticator.config.domain,
        "timestamp": datetime.now().isoformat()
    }


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    auth_header = request.headers.get("Authorization")
    
    # Skip authentication for static files and login endpoints
    if path.startswith("/resources") or path.startswith("/static") or path.startswith("/api/auth/login") or path.startswith("/api/auth/test-connection") or path.startswith("/api/health"):
        return await call_next(request)
    
    # Check for JWT token in Authorization header
    if path.startswith("/api") and auth_header:
        try:
            # Extract token from Bearer header
            token = auth_header.replace("Bearer ", "")
            user_payload = ldap_authenticator.verify_jwt_token(token)
            
            # Add user info to request state
            request.state.user = user_payload
            
            # SuperAdmin access - has access to everything including document management
            if user_payload.get('username') == 'superadmin':
                # Superadmin has full access to all routes
                pass
            
            # Admin-only routes - require admin group membership OR superadmin
            elif path.startswith("/api/indexing") or path.startswith("/api/document/add") or path.startswith("/api/document/delete") or path.startswith("/api/document/reindex"):
                user_groups = user_payload.get('groups', [])
                if not (any('admin' in group.lower() for group in user_groups) or user_payload.get('username') == 'superadmin'):
                    return JSONResponse(status_code=403, content={"detail": "Admin or SuperAdmin access required"})
            
            # Document read operations and chat - accessible to all authenticated users
            # Regular users can view, search, download documents and use chat
            
        except Exception as e:
            return JSONResponse(status_code=401, content={"detail": f"Invalid token: {str(e)}"})
    
    elif path.startswith("/api"):
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})

    return await call_next(request)


# ===== Router setup =====
app.include_router(indexing.router, prefix="/api/indexing")
app.include_router(chat_session.router, prefix="/api/chat_session")
app.include_router(document.router, prefix="/api/document")
app.include_router(chat.router, prefix="/api/chat")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Catch-all for SPA routing =====
@app.get("/{full_path:path}")
def catch_all(full_path: str):
    target = Path("static/browser") / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(Path("static/browser/index.html"))
