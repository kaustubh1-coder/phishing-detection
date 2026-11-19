import ipaddress
import re
import socket
import ssl
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from networksecurity.constants.training_pipeline import CANONICAL_FEATURE_NAMES

FEATURE_COLUMNS = CANONICAL_FEATURE_NAMES

# Categorization of all 30 canonical features
FEATURE_CATEGORIES: Dict[str, str] = {
    # Category A: Deterministically extractable from URL (pure lexical)
    "having_IP_Address": "A",
    "URL_Length": "A",
    "Shortining_Service": "A",
    "having_At_Symbol": "A",
    "double_slash_redirecting": "A",
    "Prefix_Suffix": "A",
    "having_Sub_Domain": "A",
    "port": "A",
    "HTTPS_token": "A",
    "Abnormal_URL": "A",
    "Statistical_report": "A",
    # Category B: Requires DNS/network lookup
    "DNSRecord": "B",
    "SSLfinal_State": "B",
    "Redirect": "B",
    # Category C: Requires webpage retrieval
    "Favicon": "C",
    "Request_URL": "C",
    "URL_of_Anchor": "C",
    "Links_in_tags": "C",
    "SFH": "C",
    "Submitting_to_email": "C",
    "on_mouseover": "C",
    "RightClick": "C",
    "popUpWidnow": "C",
    "Iframe": "C",
    # Category D: Requires third-party service (WHOIS)
    "Domain_registeration_length": "D",
    "age_of_domain": "D",
    # Category E: Cannot reliably be calculated in current architecture (deprecated APIs)
    "web_traffic": "E",
    "Page_Rank": "E",
    "Google_Index": "E",
    "Links_pointing_to_page": "E",
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "A": "Deterministically extractable from URL",
    "B": "Requires DNS/network lookup",
    "C": "Requires webpage retrieval",
    "D": "Requires third-party service",
    "E": "Cannot reliably be calculated in the current architecture",
}

SHORTENING_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "s.id",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
    "t.ly",
    "urlz.fr",
    "v.gd",
}


class UrlHtmlSignals(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: List[str] = []
        self.resources: List[str] = []
        self.tag_links: List[str] = []
        self.form_actions: List[str] = []
        self.favicons: List[str] = []
        self.has_iframe = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}
        tag = tag.lower()

        if tag == "a" and attr_map.get("href"):
            self.anchors.append(attr_map["href"])

        if tag in {"img", "script", "audio", "video", "source", "iframe"}:
            src = attr_map.get("src")
            if src:
                self.resources.append(src)

        if tag in {"link", "script", "meta"}:
            href = attr_map.get("href") or attr_map.get("src") or attr_map.get("content")
            if href:
                self.tag_links.append(href)

        if tag == "link" and "icon" in attr_map.get("rel", "").lower():
            href = attr_map.get("href")
            if href:
                self.favicons.append(href)

        if tag == "form":
            self.form_actions.append(attr_map.get("action", ""))

        if tag in {"iframe", "frame"}:
            self.has_iframe = True


def normalize_url(raw_url: str) -> Tuple[str, Any]:
    url = raw_url.strip()
    if not url:
        raise ValueError("Please enter a URL.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError("Please enter a valid URL with a hostname.")
    return url, parsed


def is_ip_address(hostname: str) -> bool:
    clean_host = hostname.strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        pass
    # Detect hex or octal IP representations (e.g. 0x7f.0.0.1 or 0177.0.0.1)
    if re.match(r"^(?:0x[0-9a-fA-F]+\.|\d+\.){3}(?:0x[0-9a-fA-F]+|\d+)$", clean_host):
        return True
    return False


def is_private_ip(hostname: str) -> bool:
    clean_host = hostname.strip("[]")
    try:
        ip_obj = ipaddress.ip_address(clean_host)
        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
        )
    except ValueError:
        return False


def registrable_domain(hostname: str) -> str:
    parts = [part for part in hostname.lower().split(".") if part and part != "www"]
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


def is_same_site(candidate: str, base_url: str, base_host: str) -> bool:
    if not candidate or candidate.startswith(("#", "javascript:", "mailto:", "tel:")):
        return True
    target = urlparse(urljoin(base_url, candidate))
    if not target.hostname:
        return True
    return registrable_domain(target.hostname) == registrable_domain(base_host)


def ratio_score(
    items: Iterable[str],
    base_url: str,
    base_host: str,
    safe: float,
    warn: float,
) -> float:
    """
    Computes ratio score for page elements.
    If no items exist or HTML could not be fetched, returns np.nan so that
    the fitted KNNImputer estimates this value rather than biasing towards 1 (Legitimate).
    """
    item_list = [item for item in items if item]
    if not item_list:
        return np.nan

    external = sum(1 for item in item_list if not is_same_site(item, base_url, base_host))
    ratio = external / len(item_list)
    if ratio < safe:
        return 1.0
    if ratio <= warn:
        return 0.0
    return -1.0


def check_ssl_certificate(hostname: str, port: int = 443) -> float:
    """
    Safely verifies SSL/TLS certificate with a 2.5s timeout.
    Returns 1.0 if valid, -1.0 if untrusted/expired/SSL error, 0.0 if network timeout.
    """
    if not hostname or is_ip_address(hostname) or is_private_ip(hostname):
        return -1.0

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=2.5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                if cert:
                    return 1.0
        return -1.0
    except ssl.SSLError:
        return -1.0
    except (socket.timeout, TimeoutError, OSError):
        return 0.0


def fetch_html(url: str, hostname: str) -> Tuple[str, str]:
    """
    Fetches HTML content with a 3.5s timeout.
    Skips private/loopback IP addresses immediately to prevent waiting for connection timeouts.
    """
    if is_private_ip(hostname):
        return url, ""

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NetworkSecurityURLChecker/1.0)"
        },
    )
    try:
        with urlopen(request, timeout=3.5) as response:
            content_type = response.headers.get("content-type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return response.geturl(), ""
            body = response.read(700_000).decode(charset, errors="ignore")
            return response.geturl(), body
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return url, ""


def dns_record_status(hostname: str) -> int:
    """
    Resolves domain DNS record with a controlled 2.0s timeout.
    Skips private IPs and raw IPs immediately.
    """
    if not hostname or is_ip_address(hostname) or is_private_ip(hostname):
        return -1

    orig_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(2.0)
        socket.gethostbyname(hostname)
        return 1
    except (OSError, socket.timeout, TimeoutError):
        return -1
    finally:
        socket.setdefaulttimeout(orig_timeout)


def has_non_standard_port(parsed: Any) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return True
    return port is not None and port not in {80, 443}


def extract_url_features(
    raw_url: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
    """
    Extracts the exact 30 canonical features from a raw URL.

    Design Notes:
    - Features strictly follow CANONICAL_FEATURE_NAMES.
    - Features that cannot be retrieved (e.g. unreachable HTML or unqueried WHOIS/traffic)
      are assigned np.nan so the fitted KNNImputer calculates the most probable values
      from training distribution, rather than fabricating false 'Legitimate' (1) defaults.
    - Prefix_Suffix accurately detects hyphens in the domain, in the authority before '@',
      or on raw IP hosts.
    """
    url, parsed = normalize_url(raw_url)
    hostname = parsed.hostname or ""
    host_lower = hostname.lower()
    is_ip_host = is_ip_address(hostname)

    final_url, html = fetch_html(url, hostname)
    final_host = urlparse(final_url).hostname or hostname

    parser = UrlHtmlSignals()
    if html:
        parser.feed(html)

    # 1. DNS record
    dns_status = dns_record_status(hostname)

    # 2. Prefix_Suffix check:
    # Attacker URLs often place spoofed hyphenated brands before '@' (e.g. secure-login@192.168.1.10).
    # Furthermore, raw IP hosts do not possess legitimate brand domain prefixes/suffixes.
    has_dash_in_domain = "-" in host_lower
    has_dash_in_authority = "@" in parsed.netloc and "-" in parsed.netloc.split("@")[0]
    prefix_suffix = -1 if (has_dash_in_domain or has_dash_in_authority or is_ip_host) else 1

    # 3. Subdomain count
    if is_ip_host:
        subdomain_signal = -1
    else:
        dot_count = host_lower.replace("www.", "", 1).count(".")
        subdomain_signal = 1 if dot_count <= 1 else 0 if dot_count == 2 else -1

    # 4. SSL final state
    if parsed.scheme.lower() == "https":
        ssl_state = check_ssl_certificate(hostname, parsed.port or 443)
    else:
        ssl_state = -1.0

    # 5. HTTPS token in host (e.g. http://https-secure-login.com)
    has_https_token = bool(re.search(r"(?:^|[.-])https?(?:[.-]|$)", host_lower))
    https_token = -1 if has_https_token else 1

    # 6. Redirect count and open redirect parameter detection
    has_redirect_param = bool(
        re.search(
            r"(?:redirect|url|goto|next|target|dest|destination)=https?://",
            url,
            re.IGNORECASE,
        )
    )
    redirect_count = 1 if (final_url.rstrip("/") != url.rstrip("/") or has_redirect_param) else 0

    # 7. Abnormal URL
    if is_ip_host:
        abnormal_url = -1
    elif registrable_domain(final_host) != registrable_domain(hostname):
        abnormal_url = -1
    else:
        abnormal_url = 1

    # 8. Content / HTML features (only computed if HTML was successfully fetched)
    if html:
        favicon = ratio_score(parser.favicons, url, hostname, 0.01, 0.99)
        request_url = ratio_score(parser.resources, url, hostname, 0.22, 0.61)
        url_of_anchor = ratio_score(parser.anchors, url, hostname, 0.31, 0.67)
        links_in_tags = ratio_score(parser.tag_links, url, hostname, 0.17, 0.81)

        sfh: float = np.nan
        if parser.form_actions:
            sfh = 1.0
            for action in parser.form_actions:
                if not action or action.lower() in {"about:blank", "#"}:
                    sfh = -1.0
                    break
                if not is_same_site(action, url, hostname):
                    sfh = 0.0

        submitting_email = (
            -1 if re.search(r"mailto:|mail\(", html, re.IGNORECASE) else 1
        )
        on_mouseover = (
            -1
            if re.search(r"onmouseover|window\.status", html, re.IGNORECASE)
            else 1
        )
        right_click = (
            -1
            if re.search(r"contextmenu|event\.button\s*==\s*2", html, re.IGNORECASE)
            else 1
        )
        popup_window = (
            -1 if re.search(r"window\.open|popup", html, re.IGNORECASE) else 1
        )
        iframe = -1 if parser.has_iframe else 1
    else:
        # Documented: HTML unavailable -> np.nan for KNNImputer to estimate accurately
        favicon = np.nan
        request_url = np.nan
        url_of_anchor = np.nan
        links_in_tags = np.nan
        sfh = np.nan
        submitting_email = np.nan
        on_mouseover = np.nan
        right_click = np.nan
        popup_window = np.nan
        iframe = np.nan

    # 9. Domain registration and age
    if is_ip_host:
        domain_reg_len: float = -1.0
        domain_age: float = -1.0
    else:
        # WHOIS metrics not queried via external server or unresolvable -> np.nan for KNNImputer
        domain_reg_len = np.nan
        domain_age = np.nan

    # 10. Assemble canonical feature dictionary
    features: Dict[str, Any] = {
        "having_IP_Address": -1 if is_ip_host else 1,
        "URL_Length": 1 if len(url) < 54 else 0 if len(url) <= 75 else -1,
        "Shortining_Service": -1 if registrable_domain(host_lower) in SHORTENING_DOMAINS else 1,
        "having_At_Symbol": -1 if "@" in url else 1,
        "double_slash_redirecting": -1 if url.rfind("//") > 7 else 1,
        "Prefix_Suffix": prefix_suffix,
        "having_Sub_Domain": subdomain_signal,
        "SSLfinal_State": ssl_state,
        "Domain_registeration_length": domain_reg_len,
        "Favicon": favicon,
        "port": -1 if has_non_standard_port(parsed) else 1,
        "HTTPS_token": https_token,
        "Request_URL": request_url,
        "URL_of_Anchor": url_of_anchor,
        "Links_in_tags": links_in_tags,
        "SFH": sfh,
        "Submitting_to_email": submitting_email,
        "Abnormal_URL": abnormal_url,
        "Redirect": redirect_count,
        "on_mouseover": on_mouseover,
        "RightClick": right_click,
        "popUpWidnow": popup_window,
        "Iframe": iframe,
        "age_of_domain": domain_age,
        "DNSRecord": dns_status,
        "web_traffic": np.nan,  # Unmeasured external metric -> np.nan for KNNImputer
        "Page_Rank": np.nan,  # Unmeasured external metric -> np.nan for KNNImputer
        "Google_Index": np.nan,  # Unmeasured external metric -> np.nan for KNNImputer
        "Links_pointing_to_page": np.nan,  # Unmeasured external metric -> np.nan for KNNImputer
        "Statistical_report": -1 if is_ip_host or registrable_domain(host_lower) in SHORTENING_DOMAINS else 1,
    }

    imputed_cols = [col for col in CANONICAL_FEATURE_NAMES if pd.isna(features[col])]
    unavailable_warning = None
    if not html:
        unavailable_warning = (
            "Webpage content could not be retrieved (server offline, unreachable, or access timed out). "
            f"{len(imputed_cols)} feature signals were estimated via KNN imputation from verified URL syntax. "
            "The URL was not penalized as phishing solely due to access limitations."
        )
    elif len(imputed_cols) > 0:
        unavailable_warning = (
            f"{len(imputed_cols)} external signals were safely estimated via KNN imputation without false penalties."
        )

    metadata = {
        "normalized_url": url,
        "final_url": final_url,
        "hostname": hostname,
        "html_status": "Fetched" if html else "Not fetched",
        "dns_status": "Resolved" if dns_status == 1 else "Not resolved",
        "feature_categories": FEATURE_CATEGORIES,
        "feature_availability": {
            "lexical": "available",
            "dns": "resolved" if dns_status == 1 else "unresolved_or_ip",
            "ssl": (
                "valid"
                if ssl_state == 1.0
                else ("invalid_or_http" if ssl_state == -1.0 else "unreachable")
            ),
            "html_content": "fetched" if html else "unreachable_or_offline",
            "whois": "inferred_ip" if is_ip_host else "unmeasured_third_party",
            "third_party_apis": "unmeasured_deprecated_apis",
        },
        "imputed_features_count": len(imputed_cols),
        "imputed_features": imputed_cols,
        "missing_features_count": len(imputed_cols),
        "available_features_count": len(CANONICAL_FEATURE_NAMES) - len(imputed_cols),
        "unavailable_warning": unavailable_warning,
    }

    # Strict ordering as defined by CANONICAL_FEATURE_NAMES
    ordered_values = [features[col] for col in CANONICAL_FEATURE_NAMES]
    dataframe = pd.DataFrame([ordered_values], columns=CANONICAL_FEATURE_NAMES, dtype=float)

    return dataframe, features, metadata
