"""Case-insensitive username authentication.

Django's default ``ModelBackend`` matches usernames byte-for-byte, which
trips up users who registered as ``Micu`` and later try to log in as
``micu`` (or vice versa). This backend accepts any case variant of an
existing username while still requiring an exact password match.

If two accounts somehow exist that differ only in case, we refuse the
login rather than guess — duplicate-case usernames must be resolved
manually.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend


class CaseInsensitiveModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)
        if username is None or password is None:
            return None
        try:
            user = UserModel.objects.get(**{f"{UserModel.USERNAME_FIELD}__iexact": username})
        except UserModel.DoesNotExist:
            # Run the default password hasher to mitigate timing attacks
            # that could otherwise distinguish "no such user" from "wrong password".
            UserModel().set_password(password)
            return None
        except UserModel.MultipleObjectsReturned:
            return None
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
