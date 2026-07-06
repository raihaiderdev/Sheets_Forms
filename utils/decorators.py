from functools import wraps
from flask import abort
from flask_login import current_user


def office_required(*allowed_offices):
    """Restrict a route to users whose `office` is in allowed_offices.

    Usage:
        @office_required("CEO", "DEO-SE")
        def some_view():
            ...
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated or current_user.office not in allowed_offices:
                abort(403)
            return view_func(*args, **kwargs)
        return wrapped
    return decorator
