/**
 * enrich_pubchem.js
 *
 * Legal, automated step 1 of adding a "cost" dimension to database_preprocessed.json.
 *
 * The dataset mixes two kinds of components (see each component's `component_source`):
 *   - "pubchem": small molecules identified by SMILES (amino acids, salts, sugars, DMSO...)
 *   - "uniprot": proteins/growth factors identified by a UniProt accession (e.g. P01308 = Insulin)
 * These need different lookups and, later, different vendor categories for pricing
 * (bulk biochemicals vs. recombinant proteins, which are priced per µg and much pricier).
 *
 * What it does:
 *   1. Reads all unique components from the dataset, split by component_source.
 *   2. pubchem entries -> resolved via PubChem PUG REST (official, free, ToS-compliant)
 *      using POST requests (avoids URL-encoding issues GET has with SMILES like "/" and "\").
 *      Gets: CID, IUPAC name, a friendlier common name (from synonyms), CAS number,
 *      molecular formula, molecular weight.
 *   3. uniprot entries -> resolved via UniProt REST API (official, free) to get the
 *      protein's recommended name and gene name.
 *   4. Writes a CSV template (data/price_template.csv) with an empty `price` column
 *      plus ready-to-click vendor search links, for YOU to fill in manually by
 *      browsing vendor sites normally (this is not scraping — just a convenience
 *      lookup URL). Major vendor sites (Sigma-Aldrich, Fisher, VWR, American Elements)
 *      block automated scraping (Akamai/Cloudflare bot protection) even for read-only
 *      price lookups, so this last step cannot be automated without violating their
 *      Terms of Service.
 *
 * Rate limits respected:
 *   - PubChem: max ~5 req/sec -> throttled to ~3/sec
 *   - UniProt: generous, but still throttled politely to ~3/sec
 *
 * Usage:
 *   node data/scripts/enrich_pubchem.js
 *
 * Output:
 *   data/price_template.csv
 *   data/pubchem_cache.json   (raw lookup cache, so re-runs are fast/idempotent)
 */

const fs = require('fs');
const path = require('path');

const DB_PATH = path.join(__dirname, '..', 'database_preprocessed.json');
const CACHE_PATH = path.join(__dirname, '..', 'pubchem_cache.json');
const OUT_CSV = path.join(__dirname, '..', 'price_template.csv');

const PUBCHEM_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug';
const UNIPROT_BASE = 'https://rest.uniprot.org/uniprotkb';
const THROTTLE_MS = 350; // ~3 requests/sec
const MAX_RETRIES = 3;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url, options = {}) {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const res = await fetch(url, options);
    if (res.status === 404) return null; // genuinely not found
    if (res.status === 503 || res.status === 429) {
      // transient: back off and retry
      await sleep(THROTTLE_MS * attempt * 3);
      continue;
    }
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${url}`);
    }
    return res.json();
  }
  throw new Error(`HTTP 503/429 (exhausted ${MAX_RETRIES} retries) for ${url}`);
}

const CAS_RE = /^\d{2,7}-\d{2}-\d$/;

function extractCas(synonyms) {
  return synonyms.find((s) => CAS_RE.test(s)) || '';
}

function pickCommonName(synonyms, iupacName) {
  const candidates = synonyms
    .filter((s) => !CAS_RE.test(s))
    .filter((s) => s.length <= 40)
    .sort((a, b) => a.length - b.length);
  return candidates[0] || iupacName || '';
}

async function resolvePubchem(smiles, cache) {
  if (cache[smiles]) return cache[smiles];

  const result = {
    component: smiles,
    kind: 'pubchem',
    cid: '',
    name: '',
    cas: '',
    formula: '',
    molecular_weight: '',
    error: '',
  };

  try {
    // POST avoids URL-encoding breakage GET has with "/", "\", etc. in SMILES
    await sleep(THROTTLE_MS);
    const propJson = await fetchWithRetry(
      `${PUBCHEM_BASE}/compound/smiles/property/IUPACName,MolecularFormula,MolecularWeight/JSON`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `smiles=${encodeURIComponent(smiles)}`,
      }
    );

    if (!propJson || !propJson.PropertyTable || !propJson.PropertyTable.Properties.length) {
      result.error = 'not_found_in_pubchem';
      cache[smiles] = result;
      return result;
    }

    const props = propJson.PropertyTable.Properties[0];
    result.cid = props.CID;
    result.formula = props.MolecularFormula || '';
    result.molecular_weight = props.MolecularWeight || '';
    const iupacName = props.IUPACName || '';

    await sleep(THROTTLE_MS);
    const synJson = await fetchWithRetry(`${PUBCHEM_BASE}/compound/cid/${result.cid}/synonyms/JSON`);
    const synonyms =
      synJson?.InformationList?.Information?.[0]?.Synonym ?? [];

    result.cas = extractCas(synonyms);
    result.name = pickCommonName(synonyms, iupacName);
  } catch (err) {
    result.error = err.message;
  }

  cache[smiles] = result;
  return result;
}

async function resolveUniprot(accession, cache) {
  if (cache[accession]) return cache[accession];

  const result = {
    component: accession,
    kind: 'uniprot',
    cid: '',
    name: '',
    cas: '',
    formula: '',
    molecular_weight: '',
    error: '',
  };

  try {
    await sleep(THROTTLE_MS);
    const json = await fetchWithRetry(`${UNIPROT_BASE}/${accession}.json`);
    if (!json) {
      result.error = 'not_found_in_uniprot';
      cache[accession] = result;
      return result;
    }
    result.name =
      json.proteinDescription?.recommendedName?.fullName?.value ||
      json.proteinDescription?.submissionNames?.[0]?.fullName?.value ||
      '';
    const gene = json.genes?.[0]?.geneName?.value || '';
    if (gene) result.name += ` (gene: ${gene})`;
  } catch (err) {
    result.error = err.message;
  }

  cache[accession] = result;
  return result;
}

function csvEscape(value) {
  const s = String(value ?? '');
  if (/[",\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function buildVendorSearchUrl(kind, name, cas) {
  if (kind === 'uniprot') {
    // Growth factors/cytokines are typically sold by R&D Systems / PeproTech, not Sigma.
    const q = encodeURIComponent(name || '');
    return `https://www.rndsystems.com/search?keywords=${q}`;
  }
  const q = encodeURIComponent(cas || name || '');
  return `https://www.sigmaaldrich.com/US/en/search/${q}`;
}

async function main() {
  console.log('Reading database:', DB_PATH);
  const db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));

  const freq = new Map(); // component -> {count, source}
  db.forEach((product) => {
    (product.components || []).forEach((c) => {
      const key = c.component;
      const entry = freq.get(key) || { count: 0, source: c.component_source };
      entry.count += 1;
      freq.set(key, entry);
    });
  });
  const uniqueComponents = [...freq.keys()];
  console.log(`Found ${uniqueComponents.length} unique components.`);

  let cache = {};
  if (fs.existsSync(CACHE_PATH)) {
    cache = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8'));
    console.log(`Loaded cache with ${Object.keys(cache).length} entries.`);
  }

  const results = [];
  for (let i = 0; i < uniqueComponents.length; i++) {
    const component = uniqueComponents[i];
    const { source } = freq.get(component);
    process.stdout.write(`\r[${i + 1}/${uniqueComponents.length}] Resolving (${source})...`);
    let r;
    try {
      r = source === 'uniprot' ? await resolveUniprot(component, cache) : await resolvePubchem(component, cache);
    } catch (err) {
      r = { component, kind: source, error: err.message };
      cache[component] = r;
    }
    results.push(r);
    if (i % 10 === 0) fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2));
  }
  fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2));
  console.log('\nDone resolving. Writing CSV template...');

  const header = [
    'component',
    'component_source',
    'frequency_in_dataset',
    'pubchem_cid',
    'name',
    'cas',
    'molecular_formula',
    'molecular_weight_g_per_mol',
    'price',
    'price_currency',
    'price_unit',
    'pack_size',
    'vendor',
    'vendor_product_url',
    'vendor_search_hint',
    'notes',
  ];

  const lines = [header.join(',')];
  for (const r of results) {
    const { count, source } = freq.get(r.component);
    const row = [
      r.component,
      source,
      count,
      r.cid,
      r.name,
      r.cas,
      r.formula,
      r.molecular_weight,
      '',
      '',
      '',
      '',
      '',
      '',
      buildVendorSearchUrl(source, r.name, r.cas),
      r.error || '',
    ];
    lines.push(row.map(csvEscape).join(','));
  }

  fs.writeFileSync(OUT_CSV, lines.join('\n'), 'utf-8');

  const notFound = results.filter((r) => (r.error || '').startsWith('not_found')).length;
  const errored = results.filter((r) => r.error && !r.error.startsWith('not_found')).length;
  console.log(`Wrote ${OUT_CSV}`);
  console.log(`Resolved: ${results.length - notFound - errored}/${results.length}`);
  if (notFound) console.log(`Not found: ${notFound} (check manually)`);
  if (errored) console.log(`Errors: ${errored} (re-run script to retry — cache makes it resume)`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
