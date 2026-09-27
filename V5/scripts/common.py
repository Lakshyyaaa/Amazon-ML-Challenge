"""Shared normalization utilities for the V5 pipeline."""
import os
import re
import pickle
import unicodedata

from unidecode import unidecode

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V5 = os.path.join(ROOT, 'V5')
CACHE = os.path.join(V5, 'cache')
DATA = os.path.join(ROOT, 'dataset')
os.makedirs(CACHE, exist_ok=True)

TRANSLIT_PATH = os.path.join(CACHE, 'translit_dict.pkl')

# Legal forms / generic suffixes (all countries, incl. France which is unseen in train)
LEGAL = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'llp', 'plc', 'pc', 'lp', 'pllc', 'pa', 'the', 'and', 'of', 'dba',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'scop', 'selarl', 'cie', 'et',
    'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'm', 's', 'mr', 'mrs', 'ms', 'dr',
    'smt', 'sri', 'shri', 'shree', 'public',
}

ABBR = {
    # US street types
    'st': 'street', 'str': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'dr': 'drive',
    'ln': 'lane', 'blvd': 'boulevard', 'bd': 'boulevard', 'ct': 'court', 'pl': 'place',
    'hwy': 'highway', 'pkwy': 'parkway', 'cir': 'circle', 'ter': 'terrace', 'trl': 'trail',
    'sq': 'square', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment', 'mt': 'mount',
    'ft': 'fort', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'hn': 'house',
    # French
    'r': 'rue', 'imp': 'impasse', 'all': 'allee', 'che': 'chemin', 'chem': 'chemin', 'rte': 'route',
    'pl.': 'place', 'fbg': 'faubourg', 'qu': 'quai', 'crs': 'cours',
    # India
    'rd.': 'road', 'nr': 'near', 'opp': 'opposite', 'ngr': 'nagar',
}

US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland', 'ma': 'massachusetts',
    'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri', 'mt': 'montana',
    'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico',
    'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming',
    'dc': 'district of columbia',
}
IN_STATES = {
    'tn': 'tamil nadu', 'ka': 'karnataka', 'mh': 'maharashtra', 'dl': 'delhi', 'up': 'uttar pradesh',
    'wb': 'west bengal', 'gj': 'gujarat', 'rj': 'rajasthan', 'ts': 'telangana', 'tg': 'telangana',
    'ap': 'andhra pradesh', 'kl': 'kerala', 'mp': 'madhya pradesh', 'hr': 'haryana', 'pb': 'punjab',
    'br': 'bihar', 'or': 'odisha', 'od': 'odisha', 'as': 'assam', 'jh': 'jharkhand',
    'ct': 'chhattisgarh', 'cg': 'chhattisgarh', 'uk': 'uttarakhand', 'ut': 'uttarakhand',
    'hp': 'himachal pradesh', 'ga': 'goa', 'jk': 'jammu and kashmir', 'ch': 'chandigarh',
    'py': 'puducherry', 'tr': 'tripura', 'ml': 'meghalaya', 'mn': 'manipur', 'nl': 'nagaland',
    'ar': 'arunachal pradesh', 'mz': 'mizoram', 'sk': 'sikkim',
}

TLD = {'com', 'org', 'net', 'www', 'biz', 'info', 'io'}
NULL_TOKENS = {'nan', 'null', 'none', 'n/a', 'na', '<null>', ''}

_NONLATIN = re.compile(r'[^\x00-ɏ]')
_TOKEN = re.compile(r'[a-z0-9]+')
_translit = None


def load_translit():
    global _translit
    if _translit is None:
        if os.path.exists(TRANSLIT_PATH):
            with open(TRANSLIT_PATH, 'rb') as f:
                _translit = pickle.load(f)
        else:
            _translit = {}
    return _translit


def has_nonlatin(s):
    return bool(_NONLATIN.search(s))


def to_latin(s):
    """Map native-script words through the learned dictionary, then unidecode the rest."""
    if not s:
        return ''
    if has_nonlatin(s):
        tr = load_translit()
        out = []
        for w in s.split():
            if _NONLATIN.search(w):
                key = unicodedata.normalize('NFC', w.strip('.,;:()[]{}'))
                out.append(tr.get(key) or unidecode(w))
            else:
                out.append(w)
        s = ' '.join(out)
    return unidecode(s)


def clean(s):
    """Latin, lowercase, punctuation stripped, whitespace-collapsed string."""
    if s is None or (isinstance(s, float)) or s in NULL_TOKENS:
        return ''
    s = to_latin(str(s)).lower()
    s = s.replace('&', ' and ').replace('+', ' and ')
    s = re.sub(r"[`'’]", '', s)  # o'brien -> obrien, gabriella's -> gabriellas
    toks = [t for t in _TOKEN.findall(s) if t not in NULL_TOKENS]
    return ' '.join(toks)


DOMAIN_RE = re.compile(r'([a-z0-9\-]+)\s*\.\s*(com|org|net|in|fr|co|biz|info|io)\b', re.I)
ALIAS_RE = re.compile(r'\b(?:formerly known as|formerly|f/k/a|fka|a/k/a|aka|d/b/a|d\.b\.a\.?|dba|t/a|trading as|doing business as)\b', re.I)


def name_fields(raw):
    """Return (clean_name, core_name, domain_root, alias_parts)."""
    if raw is None or isinstance(raw, float):
        return '', '', '', ''
    raw = str(raw)
    lat = to_latin(raw)
    dm = DOMAIN_RE.search(lat.lower())
    dom = dm.group(1).replace('-', '') if dm else ''
    parts = [clean(p) for p in ALIAS_RE.split(lat)]
    parts = [p for p in parts if p]
    alias = '|'.join(parts) if len(parts) > 1 else ''
    c = clean(lat)
    core = ' '.join(t for t in c.split() if t not in LEGAL and t not in TLD and len(t) > 1)
    return c, core, dom, alias


def addr_fields(raw, country):
    """Return (normalized address, house number, postal code, number tokens joined)."""
    if raw is None or isinstance(raw, float):
        return '', '', '', ''
    c = clean(raw)
    if not c:
        return '', '', '', ''
    st = US_STATES if country == 'US' else (IN_STATES if country == 'India' else {})
    toks = c.split()
    out = []
    for i, t in enumerate(toks):
        if t in st and (i >= len(toks) - 2):
            out.extend(st[t].split())
        else:
            out.append(ABBR.get(t, t))
    norm = ' '.join(out)
    nums = re.findall(r'\b(\d+)(?:[a-z]{0,2})\b', norm)
    nums_nz = [n.lstrip('0') or '0' for n in nums]
    postal = ''
    if country == 'India':
        pc = [n for n in nums if len(n) == 6 and n[0] != '0']
        postal = pc[0] if pc else ''
    elif country == 'US':
        pc = [n for n in nums if len(n) == 5]
        postal = pc[-1] if pc and len(nums) > 1 else ''
    else:
        pc = [n for n in nums if len(n) == 5]
        postal = pc[0] if pc else ''
    house = ''
    for n in nums_nz:
        if n != postal and not re.fullmatch(r'(1st|2nd|3rd)', n):
            house = n
            break
    return norm, house, postal, ' '.join(nums_nz)
