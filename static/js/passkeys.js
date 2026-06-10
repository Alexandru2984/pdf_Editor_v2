// Passkey (WebAuthn) browser flows: registration from the security page and
// password-less sign-in from the login page. External file + data-attribute
// wiring so it works under the strict CSP (no inline handlers).
//
// Uses the standard JSON round-trip helpers (parseCreationOptionsFromJSON /
// credential.toJSON()) — the server speaks exactly that wire format.
(function () {
    'use strict';

    const supported =
        typeof PublicKeyCredential !== 'undefined' &&
        typeof PublicKeyCredential.parseCreationOptionsFromJSON === 'function' &&
        typeof PublicKeyCredential.parseRequestOptionsFromJSON === 'function';

    function csrfToken() {
        const el = document.querySelector('input[name=csrfmiddlewaretoken]');
        if (el) return el.value;
        const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
        return m ? m[1] : '';
    }

    async function postJSON(url, payload) {
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken(),
            },
            body: payload ? JSON.stringify(payload) : '{}',
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
        return data;
    }

    async function fetchOptions(url) {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken() },
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.error || ('HTTP ' + resp.status));
        }
        return resp.json();
    }

    function showError(el, message) {
        const box = document.getElementById(el.getAttribute('data-passkey-error') || '');
        if (box) {
            box.textContent = message;
            box.style.display = 'block';
        } else {
            alert(message);
        }
    }

    async function registerPasskey(btn) {
        const optionsUrl = btn.getAttribute('data-passkey-register');
        const verifyUrl = btn.getAttribute('data-passkey-verify');
        const labelInput = document.getElementById(btn.getAttribute('data-passkey-label') || '');
        try {
            const optionsJSON = await fetchOptions(optionsUrl);
            const cred = await navigator.credentials.create({
                publicKey: PublicKeyCredential.parseCreationOptionsFromJSON(optionsJSON),
            });
            await postJSON(verifyUrl, {
                credential: cred.toJSON(),
                label: labelInput ? labelInput.value : '',
            });
            window.location.reload();
        } catch (err) {
            if (err.name === 'NotAllowedError') return; // user cancelled the prompt
            showError(btn, err.message || String(err));
        }
    }

    async function loginWithPasskey(btn) {
        const optionsUrl = btn.getAttribute('data-passkey-login-options');
        const verifyUrl = btn.getAttribute('data-passkey-login');
        try {
            const optionsJSON = await fetchOptions(optionsUrl);
            const cred = await navigator.credentials.get({
                publicKey: PublicKeyCredential.parseRequestOptionsFromJSON(optionsJSON),
            });
            const result = await postJSON(verifyUrl, { credential: cred.toJSON() });
            window.location.href = result.redirect || '/';
        } catch (err) {
            if (err.name === 'NotAllowedError') return;
            showError(btn, err.message || String(err));
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        // Hide passkey UI entirely on browsers without WebAuthn JSON helpers.
        if (!supported) {
            document.querySelectorAll('[data-passkey-ui]').forEach((el) => {
                el.style.display = 'none';
            });
            return;
        }
        document.addEventListener('click', (e) => {
            const reg = e.target.closest('[data-passkey-register]');
            if (reg) {
                e.preventDefault();
                registerPasskey(reg);
                return;
            }
            const log = e.target.closest('[data-passkey-login]');
            if (log) {
                e.preventDefault();
                loginWithPasskey(log);
            }
        });
    });
})();
