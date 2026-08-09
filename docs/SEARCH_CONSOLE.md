# Google Search Console runbook

The application already publishes the technical inputs Search Console needs:

- `https://pdf.micutu.com/robots.txt`
- `https://pdf.micutu.com/sitemap.xml`
- absolute canonical URLs, reciprocal `en`/`ro`/`x-default` hreflang links
- indexable localized pages and fail-closed `noindex` workflow/private pages
- structured data and a 1200×630 social preview

Search Console itself still requires an owner to add and verify the property.
That external account action is intentionally not automated by this repo.

## 1. Pre-deploy checks

Run these against the deployed release:

```bash
curl -fsS https://pdf.micutu.com/robots.txt
curl -fsS https://pdf.micutu.com/sitemap.xml
curl -fsS https://pdf.micutu.com/en/ | grep -E 'canonical|hreflang|robots'
curl -fsSI https://pdf.micutu.com/ro/
```

Confirm that `SITE_URL=https://pdf.micutu.com` in production. Canonicals and
sitemap entries deliberately use this configured origin, never the request's
`Host` header.

## 2. Verify ownership

### Recommended: Domain property through Cloudflare DNS

1. In Search Console, add a **Domain** property. Use the exact domain scope you
   want to own, without a scheme or path. `micutu.com` covers its protocols and
   subdomains; `pdf.micutu.com` limits the property to that subtree.
2. Copy the TXT record that Google provides.
3. In Cloudflare → DNS, create the TXT record using the exact name and value
   from Google's wizard. TTL `Auto` is fine.
4. Wait for DNS propagation, then click **Verify**.
5. Keep the TXT record permanently. Search Console periodically rechecks it.

Domain properties can only be verified through DNS and are the cleanest option
for protocol/subdomain coverage.

### Fallback: URL-prefix property through the HTML meta tag

Use this only for the exact `https://pdf.micutu.com/` URL-prefix property.

1. Choose **HTML tag** in Search Console.
2. From Google's tag, copy only the `content` value—not the surrounding HTML.
3. Put it in the production `.env`:

   ```dotenv
   GOOGLE_SITE_VERIFICATION=the_exact_content_value_from_google
   ```

4. Deploy normally and confirm that the unauthenticated homepage source has
   the tag inside `<head>`:

   ```bash
   curl -fsS https://pdf.micutu.com/ \
     | grep 'name="google-site-verification"'
   ```

5. Click **Verify** and keep the variable configured afterward. The token is
   not a password, but removing it can revoke verified-owner status.

Leaving `GOOGLE_SITE_VERIFICATION` empty emits no tag. DNS verification needs
no application token.

## 3. Submit discovery data

1. Open **Sitemaps** for the verified property.
2. Submit `https://pdf.micutu.com/sitemap.xml` (or `sitemap.xml` when the UI
   already supplies the property prefix).
3. Wait for status **Success** and compare the discovered URL count with the
   number of `<url>` elements returned by the live sitemap.
4. The sitemap is also advertised in `robots.txt`, so crawlers can rediscover
   it without depending on Search Console state.

## 4. Inspect representative URLs

Use **URL inspection → Test live URL** for at least:

- `https://pdf.micutu.com/en/`
- `https://pdf.micutu.com/ro/`
- `https://pdf.micutu.com/en/pdf-tools/`
- `https://pdf.micutu.com/ro/instrumente-pdf/`
- one English and one Romanian tool guide
- `https://pdf.micutu.com/en/images-to-pdf/`

Check that the page is crawlable, the user-declared canonical is itself, the
rendered HTML contains the matching language, and required CSS/images load.
Use **Request indexing** for a small representative set after a major launch;
use the sitemap for broad discovery.

The generic routes such as `/`, `/privacy/` and `/images-to-pdf/` are
compatibility/`x-default` pages and intentionally return `noindex`. Their
localized equivalents are the canonical index targets.

## 5. Monitor after launch

For the first month, review weekly:

- **Page indexing** filtered to submitted sitemap URLs;
- **Performance** by query, page, country and device;
- **Core Web Vitals** on mobile and desktop once field data exists;
- **HTTPS**, structured-data enhancements and manual/security actions;
- canonical selection and duplicate-page reasons for both languages.

Search Console data is not real-time and can take several days to appear. A
synthetic Lighthouse score is useful release evidence, but it does not replace
Search Console's field Core Web Vitals data.

## Troubleshooting

- If verification fails, fetch the public homepage in an incognito/no-cookie
  context and inspect the raw source, not the browser DOM inspector.
- If a live test is blocked at Cloudflare, inspect WAF events and narrowly
  allow Google's verified bot traffic; do not disable the WAF globally.
- If Google selects another canonical, compare redirects, status codes,
  `rel=canonical`, hreflang reciprocity, internal links and sitemap membership.
- If submitted URLs are `noindex`, verify that Search Console is testing the
  localized URL, not its generic compatibility route.
- Do not remove a DNS or meta verification token while it is still the only
  ownership method on the property.

Official references:

- [Add a property and choose its scope](https://support.google.com/webmasters/answer/34592)
- [Verify site ownership](https://support.google.com/webmasters/answer/9008080)
- [URL Inspection tool](https://support.google.com/webmasters/answer/9012289)
- [Page indexing report](https://support.google.com/webmasters/answer/7440203)
