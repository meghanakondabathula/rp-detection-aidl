# TODO: Connect dashboard.html at login button

## Task Summary
Connect the login button in login.html to redirect to dashboard.html after successful authentication.

## Implementation Plan

### Step 1: Update route.py with authentication routes
- [x] Add Flask-Login initialization and configuration
- [x] Add UserLoader callback for session management
- [x] Add `/login` route - validates credentials, logs in user, redirects to dashboard
- [x] Add `/register` route - creates new users with hashed passwords
- [x] Add `/dashboard` route - displays user dashboard with stats and predictions
- [x] Add `/logout` route - ends user session
- [x] Add `/download_dashboard` route - generates dashboard report
- [x] Add `/download_report` route - downloads individual prediction reports

### Step 2: Test the login flow
- [x] Verify login redirects to dashboard
- [x] Verify register creates new user and redirects to login/dashboard
- [x] Verify logout redirects to welcome/login
- [x] Verify dashboard displays user data correctly

## Files Modified
- `route.py` - Added all authentication and dashboard routes

## Login Flow
1. User visits `/login` → displays login.html
2. User enters credentials and clicks Login button
3. Form POSTs to `/login` route
4. Route validates credentials against database
5. On success: redirects to `/dashboard` → displays dashboard.html
6. On failure: shows error message, re-displays login form

## Notes
- Used Flask-Login for session management
- Used Werkzeug for password hashing (already configured in models.py)
- Dashboard requires: user, stats (total_tests, healthy_count, rp_count, etc.), predictions, tracking_data
- Database: SQLite at `instance/rp_detection.db`
- Secret key: 'your-secret-key-change-in-production' (should be changed in production)

